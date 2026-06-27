"""Study 1 - single-feature paired intervention; per-dimension ATE.

OFAT toggles each S/O dimension from baseline (control) to top (treatment) on a
fixed canonical base, embeds the target among distractors with counterbalanced
positions, runs trials, and estimates each dimension's ATE via matched pairs.
Also reports EI per dimension via the experimental do route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from ..analysis.ate import ate_table
from ..analysis.metrics import ei_leverage_table
from ..data.generation import make_queries
from ..experiment.runner import ExperimentRunner
from .common import assemble
from .design import ofat_pairs


@dataclass
class Study1Result:
    frame: pd.DataFrame
    ate: pd.DataFrame
    ei: pd.DataFrame


def run_study1(
    models: Sequence[str],
    per_domain: int = 1,
    domains: Optional[Sequence[str]] = None,
    prompt_styles: Sequence[str] = ("neutral",),
    seeds: Sequence[int] = (0,),
    set_size: int = 3,
    mock: Optional[bool] = None,
    progress: bool = False,
    dry_run: bool = False,
    price_in: float = 2.0,
    price_out: float = 6.0,
):
    queries = make_queries(per_domain=per_domain, domains=domains)
    points = ofat_pairs()
    q_by_id, a_by_id, sets = assemble(queries, points, set_size=set_size)

    if dry_run:
        from ..experiment.planning import compute_plan

        return compute_plan(
            "study1", q_by_id, a_by_id, sets,
            n_models=len(list(models)), n_prompt_styles=len(list(prompt_styles)),
            n_seeds=len(list(seeds)), price_in=price_in, price_out=price_out,
        )

    runner = ExperimentRunner(
        models=models, prompt_styles=prompt_styles, seeds=seeds, mock=mock
    )
    frame = runner.run(q_by_id, a_by_id, sets, progress=progress, desc="Study1 单特征干预")

    # composite pair key: pairs differ only in the toggled dimension.
    frame["pair_key"] = (
        frame["query_id"].astype(str)
        + "|" + frame["model"].astype(str)
        + "|" + frame["prompt_style"].astype(str)
        + "|" + frame["target_position"].astype(str)
        + "|" + frame["seed"].astype(str)
        + "|" + frame.get("pair_id", "").astype(str)
    )
    ate = ate_table(frame, paired_key="pair_key")
    ei = ei_leverage_table(frame, route="experimental")
    return Study1Result(frame=frame, ate=ate, ei=ei)
