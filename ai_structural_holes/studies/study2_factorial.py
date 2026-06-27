"""Study 2 - fractional factorial; main effects + key S x O interactions.

Manipulates all 8 dimensions simultaneously via a reduced 2-level design, then
estimates main effects and the priority interactions (S1xO4, S1xO2, O1xO3) with a
clustered logistic regression, plus the EI leverage table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from ..analysis.metrics import ei_leverage_table
from ..analysis.regression import logit_with_clusters
from ..codebook import all_ids
from ..data.generation import make_queries
from ..experiment.runner import ExperimentRunner
from .common import assemble
from .design import fractional_factorial

PRIORITY_INTERACTIONS = ["S1:O4", "S1:O2", "O1:O3"]


@dataclass
class Study2Result:
    frame: pd.DataFrame
    coefficients: pd.DataFrame
    ei: pd.DataFrame


def run_study2(
    models: Sequence[str],
    per_domain: int = 1,
    domains: Optional[Sequence[str]] = None,
    prompt_styles: Sequence[str] = ("neutral",),
    seeds: Sequence[int] = (0,),
    set_size: int = 3,
    n_points: Optional[int] = 16,
    interactions: Sequence[str] = tuple(PRIORITY_INTERACTIONS),
    mock: Optional[bool] = None,
    progress: bool = False,
    dry_run: bool = False,
    price_in: float = 2.0,
    price_out: float = 6.0,
):
    queries = make_queries(per_domain=per_domain, domains=domains)
    points = fractional_factorial(all_ids(), n_points=n_points)
    q_by_id, a_by_id, sets = assemble(queries, points, set_size=set_size)

    if dry_run:
        from ..experiment.planning import compute_plan

        return compute_plan(
            "study2", q_by_id, a_by_id, sets,
            n_models=len(list(models)), n_prompt_styles=len(list(prompt_styles)),
            n_seeds=len(list(seeds)), price_in=price_in, price_out=price_out,
        )

    runner = ExperimentRunner(
        models=models, prompt_styles=prompt_styles, seeds=seeds, mock=mock
    )
    frame = runner.run(q_by_id, a_by_id, sets, progress=progress, desc="Study2 分数析因")

    coefs = logit_with_clusters(
        frame, factors=all_ids(), interactions=list(interactions), cluster="query_id"
    )
    ei = ei_leverage_table(frame, route="backdoor")
    return Study2Result(frame=frame, coefficients=coefs, ei=ei)
