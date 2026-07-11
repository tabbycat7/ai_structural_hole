"""Experiment runner shared by all studies.

Given queries, an article pool, candidate sets, a model roster, prompt styles and
sampling seeds, it executes trials through the LLM client and returns a tidy
DataFrame where each row is one trial *for the target candidate*, joined with the
target's (verified) feature profile. This frame is the input to ATE / backdoor /
EI analysis.

Position counterbalancing is handled by the candidate sets passed in (see
task.protocol.build_candidate_sets with counterbalance="all_positions").
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from ..codebook import all_ids
from ..data.schema import Article, CandidateSet, Query, Trial
from ..llm.client import BaseClient, get_client
from ..llm.parallel import map_concurrent
from ..task.protocol import make_trial, trial_id_for
from .progress import DEFAULT_PROGRESS_FILE, ProgressReporter


# Substrings that mark a *permanent* provider rejection (retrying cannot help):
# the prompt itself was blocked by content moderation / risk filtering. Such a
# trial is recorded with an `api_error` marker and excluded from analysis as
# missing data, so a resumed run does not retry it forever.
_PERMANENT_API_ERROR_MARKERS = (
    "content_filter",
    "content filter",
    "high risk",
    "data_inspection_failed",
)


def _permanent_api_error(err: Exception) -> Optional[str]:
    """Return a short marker if `err` is a permanent (non-retryable) rejection."""
    status = getattr(err, "status_code", None) or getattr(err, "code", None)
    msg = str(err).lower()
    marker = next((m for m in _PERMANENT_API_ERROR_MARKERS if m in msg), None)
    if marker is not None:
        return marker.replace(" ", "_")
    # A plain 400 Bad Request is a malformed/blocked request; retrying is futile.
    if status == 400 or "badrequest" in type(err).__name__.lower():
        return "bad_request"
    return None


def trial_to_analysis_row(
    trial: Trial,
    articles_by_id: Dict[str, Article],
    queries_by_id: Dict[str, Query],
) -> dict:
    row = trial.to_row()
    target = articles_by_id[trial.candidate_set.target_id]
    q = queries_by_id.get(trial.query_id)
    if q is not None:
        row["domain"] = q.domain
    row["authenticity"] = target.authenticity
    prof = target.intended_profile
    for dim in all_ids():
        row[dim] = prof.get(dim, 0)
    for k, v in (target.meta or {}).items():
        row[k] = v
    return row


def trials_to_frame(trials: Sequence[Trial], articles_by_id: Dict[str, Article],
                    queries_by_id: Dict[str, Query]) -> pd.DataFrame:
    rows = [
        trial_to_analysis_row(t, articles_by_id, queries_by_id) for t in trials
    ]
    return pd.DataFrame(rows)


@dataclass
class ExperimentRunner:
    client: Optional[BaseClient] = None
    models: Sequence[str] = field(default_factory=list)
    prompt_styles: Sequence[str] = ("neutral",)
    seeds: Sequence[int] = (0,)
    temperature: float = 0.0
    mock: Optional[bool] = None
    use_llm_cache: Optional[bool] = None
    concurrency: int = 1
    output_mode: str = "minimal"

    def __post_init__(self):
        if self.client is None:
            self.client = get_client(mock=self.mock, use_llm_cache=self.use_llm_cache)

    def run(
        self,
        queries_by_id: Dict[str, Query],
        articles_by_id: Dict[str, Article],
        candidate_sets: Sequence[CandidateSet],
        progress: bool = False,
        desc: str = "trials",
        progress_file: Optional[Path] = None,
        on_trial: Optional[Callable[[Trial], None]] = None,
        skip: Optional[Callable[[CandidateSet, str, str, int], bool]] = None,
    ) -> pd.DataFrame:
        combos = list(
            itertools.product(candidate_sets, self.models, self.prompt_styles, self.seeds)
        )
        # Resume: drop combos already completed on a prior run so they are never
        # submitted (no API call, no cost). `skip` is supplied by the caller and
        # decides completion by whatever key it persisted (e.g. a semantic key
        # derived from columns present in trials.csv), which is robust to trial
        # id formula changes and to re-runs.
        if skip is not None:
            combos = [c for c in combos if not skip(c[0], c[1], c[2], c[3])]

        bar = None
        reporter: Optional[ProgressReporter] = None
        if progress or progress_file is not None:
            reporter = ProgressReporter(
                path=progress_file or DEFAULT_PROGRESS_FILE,
                desc=desc,
                total=len(combos),
            )
            reporter._flush(status="starting")

        if progress:
            try:
                from tqdm import tqdm

                bar = tqdm(
                    total=len(combos),
                    desc=desc,
                    unit="call",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                )
            except Exception:
                bar = None

        counter = {"cached": 0, "selected": 0}
        in_flight = {"n": 0}
        flight_lock = threading.Lock()

        def _bump_in_flight(delta: int) -> None:
            with flight_lock:
                in_flight["n"] += delta
            if reporter is not None:
                reporter.set_in_flight(in_flight["n"])

        def _run_combo(combo):
            cs, model, style, seed = combo
            q = queries_by_id[cs.query_id]
            cached_flag = {"v": False}

            def _call(**kw):
                resp = self.client.call(**kw)
                cached_flag["v"] = bool(getattr(resp, "cached", False))
                return resp

            try:
                trial = make_trial(
                    query_text=q.text,
                    domain=q.domain,
                    candidate_set=cs,
                    articles_by_id=articles_by_id,
                    model=model,
                    prompt_style=style,
                    seed=seed,
                    temperature=self.temperature,
                    call_fn=_call,
                    output_mode=self.output_mode,
                )
            except Exception as err:  # noqa: BLE001
                marker = _permanent_api_error(err)
                if marker is None:
                    # Transient failure (rate limit / network / 5xx): let it
                    # propagate so the batch stops and can be `--resume`d later.
                    raise
                # Permanent rejection: record a sentinel trial (no valid
                # decision) so it persists as done and is skipped on resume.
                trial = Trial(
                    trial_id=trial_id_for(cs, model, style, seed, self.temperature),
                    query_id=cs.query_id,
                    model=model,
                    prompt_style=style,
                    candidate_set=cs,
                    seed=seed,
                    temperature=self.temperature,
                    chosen_ids=[],
                    y={},
                    scores={},
                    rank={},
                    raw_response=str(err),
                    parse_ok=False,
                    api_error=marker,
                )
            return trial, cached_flag["v"]

        def _on_submit(_i, combo):
            _bump_in_flight(1)

        def _on_result(_i, combo, res):
            trial, cached = res
            _bump_in_flight(-1)
            if cached:
                counter["cached"] += 1
            if trial.target_y():
                counter["selected"] += 1
            cs, model, style, seed = combo
            q = queries_by_id[cs.query_id]
            if reporter is not None:
                reporter.tick(
                    cached=cached,
                    selected=bool(trial.target_y()),
                    domain=q.domain,
                    model=model,
                )
            if bar is not None:
                short_model = model.split("/")[-1][:16]
                bar.set_postfix_str(
                    f"{q.domain} {short_model} sel={counter['selected']} "
                    f"cache={counter['cached']} fly={in_flight['n']}"
                )
                bar.update(1)
            if on_trial is not None:
                on_trial(trial)

        try:
            results = map_concurrent(
                _run_combo,
                combos,
                concurrency=self.concurrency,
                on_submit=_on_submit if reporter is not None else None,
                on_result=_on_result,
            )
        except Exception:
            if reporter is not None:
                reporter.finish(status="error")
            raise
        trials = [r[0] for r in results]

        if bar is not None:
            bar.close()
        if reporter is not None:
            reporter.finish(status="done")

        return trials_to_frame(trials, articles_by_id, queries_by_id)
