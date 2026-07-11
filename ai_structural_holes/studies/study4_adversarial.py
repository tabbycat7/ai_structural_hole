"""Study 4 - reverse / adversarial: genuine vs fabricated features.

For fakeable dimensions (S1 evidence, S3 expertise), build three target versions
per query: none (baseline), genuine (verifiable), fake (fabricated but
marker-bearing). Compares selection rates to quantify deception gain, the model's
discount on fakery, and a vulnerability index; also computes Delta-EI between the
fake and genuine routes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

import pandas as pd

from ..analysis.metrics import deception_gain
from ..causal.ei import ei_from_do_table
from ..codebook import baseline_profile, get_dimension
from ..data.generation import make_article
from ..data.schema import Article, CandidateSet
from ..experiment.runner import ExperimentRunner
from ..task.protocol import build_candidate_sets
from .common import get_queries, make_distractor_pool, make_gen_client, resolve_route, run_deferred_jobs

if TYPE_CHECKING:
    from ..study_output import StudyModelSink

FAKEABLE_DIMS = ["S1", "S3"]


@dataclass
class Study4Result:
    frame: pd.DataFrame
    deception: pd.DataFrame
    delta_ei: pd.DataFrame


def _build(queries, fakeable_dims, set_size, seed,
           route="template", gen_client=None, gen_model=None,
           distractor_route=None, progress=False, concurrency=1,
           use_variant_store=True):
    import random

    rng = random.Random(seed)
    q_by_id = {q.id: q for q in queries}
    a_by_id = {}
    sets = []

    base_texts = {}
    if route == "llm":
        from ..data.base_articles import load_base_texts

        base_texts = load_base_texts()

    pool_passages = {}
    if distractor_route == "real":
        from ..data.query_pool import load_pool_passages

        pool_passages = load_pool_passages()

    jobs: list = []
    for q in queries:
        q_route, base_text = resolve_route(q, route, gen_client, base_texts)
        d_route = distractor_route or q_route
        if d_route == "llm" and (gen_client is None or not base_text):
            d_route = "template"
        distractors = make_distractor_pool(
            q, n=4, rng=rng,
            route=d_route, client=gen_client, gen_model=gen_model, base_text=base_text,
            passages=pool_passages.get(q.id),
            defer=jobs,
        )
        for d in distractors:
            a_by_id[d.id] = d
        # Draw competitors ONCE per query and reuse them for every target version
        # (all dims, none/genuine/fake). This keeps the competition environment
        # identical across the three variants, so deception metrics reflect only
        # the target's own authenticity, not a change in who it competes against.
        shared_distractors = rng.sample(list(distractors), set_size - 1)
        for dim in fakeable_dims:
            top = get_dimension(dim).top_code()
            variants = {
                "none": (baseline_profile(), "genuine"),
                "genuine": ({**baseline_profile(), dim: top}, "genuine"),
                "fake": ({**baseline_profile(), dim: top}, "fake"),
            }
            for vname, (prof, auth) in variants.items():
                tgt = make_article(
                    q, prof, is_target=True, authenticity=auth,
                    suffix=f"{dim}-{vname}",
                    meta={"fakeable_dim": dim, "variant": vname},
                    route=q_route, client=gen_client, gen_model=gen_model,
                    base_text=base_text, defer=jobs,
                )
                a_by_id[tgt.id] = tgt
                sets.extend(
                    build_candidate_sets(
                        tgt, distractors, set_size=set_size,
                        counterbalance="all_positions", rng=rng,
                        fixed_distractors=shared_distractors,
                    )
                )
    run_deferred_jobs(
        jobs, q_by_id, concurrency=concurrency, progress=progress,
        use_variant_store=use_variant_store,
    )
    return q_by_id, a_by_id, sets


def run_study4(
    models: Sequence[str],
    per_domain: int = 1,
    domains: Optional[Sequence[str]] = None,
    fakeable_dims: Sequence[str] = tuple(FAKEABLE_DIMS),
    prompt_styles: Sequence[str] = ("neutral", "critical_eval"),
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
    gen_client = make_gen_client(gen_route, mock, dry_run)
    q_by_id, a_by_id, sets = _build(
        queries, list(fakeable_dims), set_size, seed=0,
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
            "study4", q_by_id, a_by_id, sets,
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
        q_by_id, a_by_id, sets, progress=progress, desc="Study4 反向对抗",
        progress_file=progress_file,
        on_trial=output_sink.on_trial if output_sink else None,
    )

    # deception metrics per fakeable dimension
    dec_rows = []
    delta_rows = []
    for dim in fakeable_dims:
        sub = frame[frame.get("fakeable_dim", "") == dim]
        if sub.empty:
            continue
        dec_rows.append(deception_gain(sub, dim))

        # Delta-EI: EI of fake route vs genuine route (P(Y|do(dim)) per route)
        gen_df = sub[sub["variant"].isin(["none", "genuine"])]
        fake_df = sub[sub["variant"].isin(["none", "fake"])]

        def _ei_for(d):
            p0 = d[d["variant"] == "none"]["y"].mean()
            p1 = d[d["variant"] != "none"]["y"].mean()
            if pd.isna(p0) or pd.isna(p1):
                return float("nan")
            return ei_from_do_table({0: float(p0), 1: float(p1)}).ei

        ei_gen = _ei_for(gen_df)
        ei_fake = _ei_for(fake_df)
        delta_rows.append({
            "factor": dim,
            "EI_genuine": ei_gen,
            "EI_fake": ei_fake,
            "delta_EI": (ei_fake - ei_gen) if not (pd.isna(ei_fake) or pd.isna(ei_gen)) else float("nan"),
        })

    return Study4Result(
        frame=frame,
        deception=pd.DataFrame(dec_rows),
        delta_ei=pd.DataFrame(delta_rows),
    )


def build_study4_materials(
    models: Sequence[str],
    per_domain: int = 1,
    domains: Optional[Sequence[str]] = None,
    fakeable_dims: Sequence[str] = tuple(FAKEABLE_DIMS),
    set_size: int = 3,
    mock: Optional[bool] = None,
    progress: bool = False,
    gen_route: str = "llm",
    gen_model: Optional[str] = None,
    query_source: str = "builtin",
    distractor_route: Optional[str] = None,
    concurrency: int = 1,
    use_variant_store: bool = True,
) -> pd.DataFrame:
    """Generate + freeze the S1/S3 none/genuine/fake target articles only.

    Runs no model selection: it builds (and, on the llm route, freezes) the
    deception target articles, then returns an audit sheet listing each article's
    text plus blank columns for a human reviewer to fill in. Because target ids
    are deterministic (see `make_article`/`stable_id`), a later `study4` run with
    the same parameters reuses these exact verified texts via the variant store.
    """
    queries = get_queries(query_source, per_domain=per_domain, domains=domains)
    gen_client = make_gen_client(gen_route, mock, dry_run=False)
    q_by_id, a_by_id, _sets = _build(
        queries, list(fakeable_dims), set_size, seed=0,
        route=gen_route if gen_client else "template",
        gen_client=gen_client, gen_model=gen_model or (list(models)[0] if models else None),
        distractor_route=distractor_route,
        progress=progress,
        concurrency=concurrency,
        use_variant_store=use_variant_store,
    )

    rows = []
    for art in a_by_id.values():
        if not art.is_target:
            continue
        meta = art.meta or {}
        dim = meta.get("fakeable_dim")
        if not dim:
            continue
        q = q_by_id.get(art.query_id)
        rows.append({
            "article_id": art.id,
            "query_id": art.query_id,
            "domain": q.domain if q is not None else "",
            "dim": dim,
            "variant": meta.get("variant", ""),
            "generator": meta.get("generator", ""),
            "n_chars": art.n_chars,
            "text": art.text,
            # blank columns for the human reviewer
            "verifiable": "",
            "source_url_or_doi": "",
            "verdict": "",
            "reviewer": "",
            "note": "",
        })

    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit = audit.sort_values(["query_id", "dim", "variant"]).reset_index(drop=True)
    return audit
