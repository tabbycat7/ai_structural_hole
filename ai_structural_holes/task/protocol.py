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
from .prompts import LETTERS, build_messages


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
) -> List[CandidateSet]:
    """Return candidate sets that place `target` at controlled positions.

    counterbalance:
      - "all_positions": one set per possible target position (0..set_size-1).
      - "single": one set, target at a random position.
    Distractors are sampled (without replacement) to fill the remaining slots.
    """
    rng = rng or random.Random(0)
    if set_size < 2:
        raise ValueError("set_size must be >= 2")
    n_distract = set_size - 1
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
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    candidates = [text]
    match = _JSON_RE.search(text)
    if match:
        candidates.append(match.group())
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
# Trial assembly
# --------------------------------------------------------------------------- #
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
) -> Trial:
    """Run one trial end-to-end via `call_fn(model, messages, temperature, seed)`.

    `call_fn` must return an object with a `.text` attribute (see llm client).
    This keeps the protocol independent of the transport (real API or mock).
    """
    ordered_ids = candidate_set.ordered_ids
    candidate_texts = [articles_by_id[cid].text for cid in ordered_ids]
    messages, _letters = build_messages(query_text, candidate_texts, prompt_style, domain)

    resp = call_fn(model=model, messages=messages, temperature=temperature, seed=seed)
    raw = getattr(resp, "text", str(resp))
    parsed = parse_decision(raw, ordered_ids)

    trial_id = stable_id(candidate_set.query_id, ordered_ids, model, prompt_style, seed, temperature)
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
