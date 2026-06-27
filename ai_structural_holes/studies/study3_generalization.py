"""Study 3 - generalization across models, domains, prompts, and R.

Repeats the OFAT design over the full M x domain x prompt x position grid and
checks stability / ranking agreement of effects across conditions: a per-stratum
EI leverage table plus cross-model concordance (Kendall's W, Spearman).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from ..analysis.metrics import cross_model_consistency, ei_leverage_table, position_adjusted_rate
from ..config import DOMAINS, PROMPT_STYLES
from ..data.generation import make_queries
from ..experiment.runner import ExperimentRunner
from .common import assemble
from .design import ofat_pairs


@dataclass
class Study3Result:
    frame: pd.DataFrame
    ei_by_model: pd.DataFrame
    ei_by_domain: pd.DataFrame
    consistency: dict
    position_rate: pd.DataFrame


def _ei_by(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby(col):
        tab = ei_leverage_table(g, route="experimental")
        tab[col] = key
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_study3(
    models: Sequence[str],
    per_domain: int = 1,
    domains: Optional[Sequence[str]] = None,
    prompt_styles: Sequence[str] = tuple(PROMPT_STYLES),
    seeds: Sequence[int] = (0,),
    set_size: int = 3,
    mock: Optional[bool] = None,
    progress: bool = False,
    dry_run: bool = False,
    price_in: float = 2.0,
    price_out: float = 6.0,
):
    domains = list(domains or DOMAINS)
    queries = make_queries(per_domain=per_domain, domains=domains)
    points = ofat_pairs()
    q_by_id, a_by_id, sets = assemble(queries, points, set_size=set_size)

    if dry_run:
        from ..experiment.planning import compute_plan

        return compute_plan(
            "study3", q_by_id, a_by_id, sets,
            n_models=len(list(models)), n_prompt_styles=len(list(prompt_styles)),
            n_seeds=len(list(seeds)), price_in=price_in, price_out=price_out,
        )

    runner = ExperimentRunner(
        models=models, prompt_styles=prompt_styles, seeds=seeds, mock=mock
    )
    frame = runner.run(q_by_id, a_by_id, sets, progress=progress, desc="Study3 泛化矩阵")

    ei_by_model = _ei_by(frame, "model")
    ei_by_domain = _ei_by(frame, "domain") if "domain" in frame.columns else pd.DataFrame()
    consistency = cross_model_consistency(frame)
    pos_rate = position_adjusted_rate(frame, by=["model"])
    return Study3Result(
        frame=frame,
        ei_by_model=ei_by_model,
        ei_by_domain=ei_by_domain,
        consistency=consistency,
        position_rate=pos_rate,
    )
