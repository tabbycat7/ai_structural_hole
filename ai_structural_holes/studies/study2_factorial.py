"""Study 2 - fractional factorial; main effects + key S x O interactions.

Manipulates all 8 dimensions simultaneously via a reduced 2-level design, then
estimates main effects and the priority interactions (S1xO4, S1xO2, O1xO3) with a
clustered logistic regression, plus the EI leverage table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

import pandas as pd

from ..analysis.metrics import ei_leverage_table
from ..analysis.regression import logit_with_clusters
from ..codebook import all_ids
from ..experiment.runner import ExperimentRunner
from .common import assemble, get_queries, make_gen_client
from .design import fractional_factorial

if TYPE_CHECKING:
    from ..study_output import StudyModelSink

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
    gen_route: str = "template",
    gen_model: Optional[str] = None,
    query_source: str = "builtin",
    distractor_route: Optional[str] = None,
    concurrency: int = 1,
    use_variant_store: bool = True,
    output_mode: str = "minimal",
    progress_file=None,
    output_sink: Optional["StudyModelSink"] = None,
    use_llm_cache: Optional[bool] = None,
):
    queries = get_queries(query_source, per_domain=per_domain, domains=domains)
    points = fractional_factorial(all_ids(), n_points=n_points)
    gen_client = make_gen_client(gen_route, mock, dry_run)
    q_by_id, a_by_id, sets = assemble(
        queries, points, set_size=set_size,
        route=gen_route if gen_client else "template",
        gen_client=gen_client, gen_model=gen_model or (list(models)[0] if models else None),
        distractor_route=distractor_route,
        progress=progress,
        concurrency=concurrency,
        use_variant_store=use_variant_store,
    )

    if dry_run:
        from ..experiment.planning import compute_plan

        return compute_plan(
            "study2", q_by_id, a_by_id, sets,
            n_models=len(list(models)), n_prompt_styles=len(list(prompt_styles)),
            n_seeds=len(list(seeds)), price_in=price_in, price_out=price_out,
            output_mode=output_mode,
        )

    runner = ExperimentRunner(
        models=models, prompt_styles=prompt_styles, seeds=seeds, mock=mock,
        concurrency=concurrency, output_mode=output_mode, use_llm_cache=use_llm_cache,
    )
    if output_sink is not None:
        output_sink.bind_context(a_by_id, q_by_id)
    frame = runner.run(
        q_by_id, a_by_id, sets, progress=progress, desc="Study2 分数析因",
        progress_file=progress_file,
        on_trial=output_sink.on_trial if output_sink else None,
    )

    coefs = logit_with_clusters(
        frame, factors=all_ids(), interactions=list(interactions), cluster="query_id"
    )
    ei = ei_leverage_table(frame, route="backdoor")
    return Study2Result(frame=frame, coefficients=coefs, ei=ei)
