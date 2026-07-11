"""Study 1 - single-feature paired intervention; per-dimension ATE.

OFAT toggles each S/O dimension from baseline (control) to top (treatment) on a
fixed canonical base, embeds the target among distractors with counterbalanced
positions, runs trials, and estimates each dimension's ATE via matched pairs.
Also reports EI per dimension via the experimental do route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

import pandas as pd

from ..analysis.ate import ate_table
from ..analysis.metrics import ei_leverage_table
from ..experiment.runner import ExperimentRunner
from .common import assemble, get_queries, make_gen_client
from .design import ofat_pairs

if TYPE_CHECKING:
    from ..study_output import StudyModelSink


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
    points = ofat_pairs()
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
            "study1", q_by_id, a_by_id, sets,
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
        q_by_id, a_by_id, sets, progress=progress, desc="Study1 单特征干预",
        progress_file=progress_file,
        on_trial=output_sink.on_trial if output_sink else None,
    )

    # composite pair key: pairs differ only in the toggled dimension.
    frame["pair_key"] = (
        frame["query_id"].astype(str)
        + "|" + frame["model"].astype(str)
        + "|" + frame["prompt_style"].astype(str)
        + "|" + frame["target_position"].astype(str)
        + "|" + frame["seed"].astype(str)
        + "|" + frame.get("pair_id", "").astype(str)
    )

    # Parse failures are missing outcomes, not confirmed non-selections: drop
    # them from the estimand rather than coding them as Y=0. The full frame
    # (incl. failures) is still returned/persisted so the parse rate is auditable.
    analysis_df = frame
    if "parse_ok" in frame.columns:
        analysis_df = frame[frame["parse_ok"] == 1].copy()

    # Confine each factor's contrast to its own OFAT pairs (target_dim), and
    # cluster the paired-bootstrap CI on the query (the true independent unit).
    ate = ate_table(analysis_df, paired_key="pair_key", cluster="query_id")
    ei = ei_leverage_table(analysis_df, route="experimental", scope_col="target_dim")
    return Study1Result(frame=frame, ate=ate, ei=ei)
