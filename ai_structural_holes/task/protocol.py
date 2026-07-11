"""RAG multi-candidate competition protocol.

Responsibilities:
  - Build candidate sets (target + distractors) of a given size and competitor
    quality -> encodes R.
  - Counterbalance the target's position across trials (full permutation for
    small sets, otherwise systematic rotation / Latin-square style) so position
    bias in R is controllable.
  - Parse the model's JSON response into per-candidate Y, ranks, and scores.

Parsing is defensive: malformed output is flagged (parse_ok=False) rather than
crashing the run.
"""
from __future__ import annotations

import itertools
import json
import random
import re
from typing import Dict, List, Optional, Sequence

from ..data.schema import Article, CandidateSet, Trial, stable_id
from .prompts import LETTERS, build_messages, max_tokens_for_mode


# --------------------------------------------------------------------------- #
# Candidate-set construction (R)
# --------------------------------------------------------------------------- #
def build_candidate_sets(
    target: Article,
    distractors: Sequence[Article],
    set_size: int,
    competitor_quality: str = "mixed",
    counterbalance: str = "all_positions",
    rng: Optional[random.Random] = None,
    fixed_distractors: Optional[Sequence[Article]] = None,
) -> List[CandidateSet]:
    """Return candidate sets that place `target` at controlled positions.

    counterbalance:
      - "all_positions": one set per possible target position (0..set_size-1).
      - "single": one set, target at a random position.

    `fixed_distractors`: if given, these exact competitors are used verbatim
    (no sampling), so every target built with the same list faces an identical
    competition environment. This is what makes a matched (control, treatment)
    pair differ *only* in the target's own text. Otherwise `set_size-1`
    distractors are sampled without replacement from `distractors`.
    """
    rng = rng or random.Random(0)
    if set_size < 2:
        raise ValueError("set_size must be >= 2")
    n_distract = set_size - 1
    if fixed_distractors is not None:
        if len(fixed_distractors) < n_distract:
            raise ValueError(
                f"need >={n_distract} fixed_distractors, got {len(fixed_distractors)}"
            )
        chosen = list(fixed_distractors)[:n_distract]
    else:
        if len(distractors) < n_distract:
            raise ValueError(
                f"need >={n_distract} distractors, got {len(distractors)}"
            )
        chosen = rng.sample(list(distractors), n_distract)
    sets: List[CandidateSet] = []

    positions = (
        range(set_size) if counterbalance == "all_positions" else [rng.randrange(set_size)]
    )
    for pos in positions:
        ordered: List[str] = [d.id for d in chosen]
        ordered.insert(pos, target.id)
        sets.append(
            CandidateSet(
                query_id=target.query_id,
                ordered_ids=ordered,
                target_id=target.id,
                competitor_quality=competitor_quality,
            )
        )
    return sets


# --------------------------------------------------------------------------- #
# Response parsing -> Y / ranks / scores
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _balanced_objects(text: str) -> List[str]:
    """Return every top-level {...} block via brace-depth scanning.

    Robust to models that wrap the JSON in prose or emit several objects; string
    literals (and their escapes) are respected so braces inside strings don't
    throw off the depth count.
    """
    objs: List[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objs.append(text[start : i + 1])
    return objs


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    candidates: List[str] = [text.strip()]
    # Peel markdown code fences (```json ... ```), which many models add despite
    # the instruction not to.
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    # Balanced-brace blocks; prefer the last complete object (models sometimes
    # emit a reasoning object then the final answer).
    for block in reversed(_balanced_objects(text)):
        candidates.append(block)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def parse_decision(
    raw_response: str,
    ordered_ids: List[str],
) -> Dict[str, object]:
    """Map a raw model response to per-candidate decisions.

    Letters in the response refer to positions: A -> ordered_ids[0], etc.
    Returns dict with keys: y, scores, rank, chosen_ids, parse_ok.
    """
    letters = LETTERS[: len(ordered_ids)]
    letter_to_id = {ltr: cid for ltr, cid in zip(letters, ordered_ids)}

    y = {cid: 0 for cid in ordered_ids}
    scores: Dict[str, float] = {}
    rank: Dict[str, int] = {}
    chosen_ids: List[str] = []

    data = _extract_json(raw_response)
    parse_ok = data is not None

    if data:
        # Citation task (Study 5): a `cited` list marks every candidate the model
        # used, so multiple candidates can have Y=1. An empty list is a valid
        # "cited nothing" outcome; a non-empty list with no resolvable letters is
        # a parse failure.
        if isinstance(data.get("cited"), list):
            raw_cited = data["cited"]
            for ltr in raw_cited:
                ltr = str(ltr).strip().upper()[:1]
                if ltr in letter_to_id:
                    cid = letter_to_id[ltr]
                    if y[cid] == 0:
                        y[cid] = 1
                        chosen_ids.append(cid)
            if raw_cited and not chosen_ids:
                parse_ok = False
        else:
            choice = str(data.get("choice", "")).strip().upper()[:1]
            if choice in letter_to_id:
                cid = letter_to_id[choice]
                y[cid] = 1
                chosen_ids = [cid]
            else:
                parse_ok = False

        raw_scores = data.get("scores", {}) or {}
        if isinstance(raw_scores, dict):
            for ltr, val in raw_scores.items():
                ltr = str(ltr).strip().upper()[:1]
                if ltr in letter_to_id:
                    try:
                        scores[letter_to_id[ltr]] = float(val)
                    except Exception:
                        pass

        raw_rank = data.get("ranking", []) or []
        if isinstance(raw_rank, list):
            for pos, ltr in enumerate(raw_rank):
                ltr = str(ltr).strip().upper()[:1]
                if ltr in letter_to_id:
                    rank[letter_to_id[ltr]] = pos

    return {
        "y": y,
        "scores": scores,
        "rank": rank,
        "chosen_ids": chosen_ids,
        "parse_ok": parse_ok,
    }


# --------------------------------------------------------------------------- #
# Query-rewrite parsing (Study 6)
# --------------------------------------------------------------------------- #
def parse_rewrite(
    raw_response: str,
    fallback_query: str,
    max_queries: int = 8,
) -> Dict[str, object]:
    """Parse a `{"queries": [...]}` rewrite response into a query list.

    Returns dict with keys: queries (non-empty list of stripped strings) and
    parse_ok. On malformed output or an empty list, falls back to
    `[fallback_query]` and marks parse_ok=False so the run stays defined and the
    failure is auditable.
    """
    data = _extract_json(raw_response)
    queries: List[str] = []
    if data is not None:
        raw_queries = data.get("queries")
        if isinstance(raw_queries, list):
            seen: set = set()
            for q in raw_queries:
                q = str(q).strip()
                if q and q not in seen:
                    seen.add(q)
                    queries.append(q)
                if len(queries) >= max_queries:
                    break

    parse_ok = bool(queries)
    if not queries:
        queries = [fallback_query]
    return {"queries": queries, "parse_ok": parse_ok}


# --------------------------------------------------------------------------- #
# Trial assembly
# --------------------------------------------------------------------------- #
def trial_id_for(
    candidate_set: CandidateSet,
    model: str,
    prompt_style: str,
    seed: int,
    temperature: float,
) -> str:
    """Deterministic trial id for a (candidate set, model, style, seed, temp).

    Single source of truth shared by `make_trial` and the runner's resume /
    skip logic so a re-run recomputes exactly the id that was written to disk.

    `target_id` is part of the key: in Study 6's real end-to-end RAG, several
    *absent* targets of the same query share the identical top-k `ordered_ids`,
    so without it their ids would collide and resume could skip un-run targets.
    """
    return stable_id(
        candidate_set.query_id,
        candidate_set.ordered_ids,
        candidate_set.target_id,
        model,
        prompt_style,
        seed,
        temperature,
    )


def make_trial(
    *,
    query_text: str,
    domain: str,
    candidate_set: CandidateSet,
    articles_by_id: Dict[str, Article],
    model: str,
    prompt_style: str,
    seed: int,
    temperature: float,
    call_fn,
    output_mode: str = "minimal",
) -> Trial:
    """Run one trial end-to-end via `call_fn(model, messages, temperature, seed)`.

    `call_fn` must return an object with a `.text` attribute (see llm client).
    This keeps the protocol independent of the transport (real API or mock).
    """
    ordered_ids = candidate_set.ordered_ids
    candidate_texts = [articles_by_id[cid].text for cid in ordered_ids]
    messages, _letters = build_messages(
        query_text, candidate_texts, prompt_style, domain, output_mode=output_mode,
    )

    resp = call_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens_for_mode(output_mode),
    )
    raw = getattr(resp, "text", str(resp))
    parsed = parse_decision(raw, ordered_ids)

    trial_id = trial_id_for(candidate_set, model, prompt_style, seed, temperature)
    return Trial(
        trial_id=trial_id,
        query_id=candidate_set.query_id,
        model=model,
        prompt_style=prompt_style,
        candidate_set=candidate_set,
        seed=seed,
        temperature=temperature,
        chosen_ids=parsed["chosen_ids"],
        y=parsed["y"],
        scores=parsed["scores"],
        rank=parsed["rank"],
        raw_response=raw,
        parse_ok=parsed["parse_ok"],
    )


def all_position_permutations(ids: List[str], max_perms: int = 24) -> List[List[str]]:
    """Full permutations for small sets; capped sample for larger ones."""
    perms = list(itertools.permutations(ids))
    if len(perms) <= max_perms:
        return [list(p) for p in perms]
    rng = random.Random(0)
    return [list(rng.sample(ids, len(ids))) for _ in range(max_perms)]
