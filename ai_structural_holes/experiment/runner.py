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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..codebook import all_ids
from ..data.schema import Article, CandidateSet, Query, Trial
from ..llm.client import BaseClient, get_client
from ..task.protocol import make_trial


def trials_to_frame(trials: Sequence[Trial], articles_by_id: Dict[str, Article],
                    queries_by_id: Dict[str, Query]) -> pd.DataFrame:
    rows = []
    for t in trials:
        row = t.to_row()
        target = articles_by_id[t.candidate_set.target_id]
        q = queries_by_id.get(t.query_id)
        if q is not None:
            row["domain"] = q.domain
        row["authenticity"] = target.authenticity
        prof = target.profile
        for dim in all_ids():
            row[dim] = prof.get(dim, 0)
        for k, v in (target.meta or {}).items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class ExperimentRunner:
    client: Optional[BaseClient] = None
    models: Sequence[str] = field(default_factory=list)
    prompt_styles: Sequence[str] = ("neutral",)
    seeds: Sequence[int] = (0,)
    temperature: float = 0.0
    mock: Optional[bool] = None

    def __post_init__(self):
        if self.client is None:
            self.client = get_client(mock=self.mock)

    def run(
        self,
        queries_by_id: Dict[str, Query],
        articles_by_id: Dict[str, Article],
        candidate_sets: Sequence[CandidateSet],
        progress: bool = False,
        desc: str = "trials",
    ) -> pd.DataFrame:
        trials: List[Trial] = []
        combos = list(
            itertools.product(candidate_sets, self.models, self.prompt_styles, self.seeds)
        )

        bar = None
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

        counter = {"cached": 0}

        def _counting_call(**kw):
            resp = self.client.call(**kw)
            if getattr(resp, "cached", False):
                counter["cached"] += 1
            return resp

        for cs, model, style, seed in combos:
            q = queries_by_id[cs.query_id]
            trial = make_trial(
                query_text=q.text,
                domain=q.domain,
                candidate_set=cs,
                articles_by_id=articles_by_id,
                model=model,
                prompt_style=style,
                seed=seed,
                temperature=self.temperature,
                call_fn=_counting_call,
            )
            trials.append(trial)
            if bar is not None:
                short_model = model.split("/")[-1][:18]
                bar.set_postfix_str(f"model={short_model} cached={counter['cached']}")
                bar.update(1)

        if bar is not None:
            bar.close()

        return trials_to_frame(trials, articles_by_id, queries_by_id)
