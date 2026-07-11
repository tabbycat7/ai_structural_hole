"""Study 3 - generalization across models, domains, prompts, and R.

Repeats the OFAT design over the full M x domain x prompt x position grid and
checks stability / ranking agreement of effects across conditions: a per-stratum
EI leverage table plus cross-model concordance (Kendall's W, Spearman).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

import pandas as pd

from ..analysis.metrics import cross_model_consistency, ei_leverage_table, position_adjusted_rate
from ..config import DOMAINS, PROMPT_STYLES
from ..experiment.runner import ExperimentRunner
from .common import assemble, get_queries, make_gen_client
from .design import ofat_pairs

if TYPE_CHECKING:
    from ..study_output import StudyModelSink


@dataclass
class Study3Result:
    frame: pd.DataFrame
    ei_by_model: pd.DataFrame
    ei_by_domain: pd.DataFrame
    consistency: dict
    position_rate: pd.DataFrame


def _ei_by(frame: pd.DataFrame, col: str, scope_col: str = "target_dim") -> pd.DataFrame:
    """Per-stratum EI leverage table over an OFAT frame.

    Confines each factor's contrast to its own OFAT pairs via `scope_col`
    (matching Study 1); otherwise a factor pinned high as a fixed baseline
    inside another dimension's pairs would contaminate its EI/ATE.
    """
    rows = []
    for key, g in frame.groupby(col):
        tab = ei_leverage_table(g, route="experimental", scope_col=scope_col)
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
    domains = list(domains or DOMAINS)
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
            "study3", q_by_id, a_by_id, sets,
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
        q_by_id, a_by_id, sets, progress=progress, desc="Study3 泛化矩阵",
        progress_file=progress_file,
        on_trial=output_sink.on_trial if output_sink else None,
    )

    # Parse failures are missing outcomes, not confirmed non-selections: drop them
    # from the estimands (matching Study 1). The full frame is still returned so
    # the parse rate stays auditable.
    analysis_df = frame
    if "parse_ok" in frame.columns:
        analysis_df = frame[frame["parse_ok"] == 1].copy()

    # OFAT design: scope each factor's EI to its own target_dim pairs.
    ei_by_model = _ei_by(analysis_df, "model", scope_col="target_dim")
    ei_by_domain = (
        _ei_by(analysis_df, "domain", scope_col="target_dim")
        if "domain" in analysis_df.columns else pd.DataFrame()
    )
    consistency = cross_model_consistency(
        analysis_df, route="experimental", scope_col="target_dim"
    )
    pos_rate = position_adjusted_rate(analysis_df, by=["model"])
    return Study3Result(
        frame=frame,
        ei_by_model=ei_by_model,
        ei_by_domain=ei_by_domain,
        consistency=consistency,
        position_rate=pos_rate,
    )
