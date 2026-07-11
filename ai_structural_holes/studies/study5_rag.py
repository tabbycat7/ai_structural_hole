"""Study 5 - real RAG retrieval loop (Scheme A).

Upgrades the forced "3-inlined-candidates single choice" into a real two-stage
RAG pipeline and decomposes the structural-hole effect into two channels:

  - Retrieval channel  : does a controlled target feature raise the probability
    that the target is retrieved into the top-k of a *real* corpus?
      P_retrieve = P(target in top-k)
  - Generation channel : once the target is in the context, does the feature
    raise the probability the model actually cites it (forced inclusion isolates
    this from retrieval)?
      P_cite|ctx = P(target cited | target in context)
  - End-to-end         : P_e2e ~= P_retrieve * P_cite|ctx.

Design reuse:
  - Target articles are the *frozen* Study 1 LLM variants: we rebuild the target
    shells from `ofat_pairs()` (ids are route-independent) and overlay the frozen
    text from `data/variant_articles/` via `apply_record` — no LLM call, no
    network, no cost.
  - The generation stage reuses `ExperimentRunner` with `output_mode="cite"`.
  - Competitors are real corpus passages (see `retrieval.corpus`), identical for
    a target's control and treatment, so the paired contrast is clean.
"""
from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..study_output import StudyModelSink

from ..analysis.ate import ate_table
from ..analysis.metrics import ei_leverage_table
from ..codebook import all_ids, get_dimension
from ..config import DOMAINS, RAG_HYBRID_ALPHA, RAG_RETRIEVER, RAG_TOP_K
from ..data.schema import Article, CandidateSet
from ..data.variant_articles import apply_record, load_variant_store
from ..experiment.runner import ExperimentRunner
from ..retrieval.corpus import CorpusDoc, load_corpus
from ..retrieval.retriever import HybridRetriever
from ..task.protocol import build_candidate_sets
from .common import build_targets, get_queries, make_progress_bar
from .design import ofat_pairs


@dataclass
class Study5Result:
    retrieval_frame: pd.DataFrame
    gen_frame: pd.DataFrame
    ate_retrieved: pd.DataFrame
    ate_cite: pd.DataFrame
    ei: pd.DataFrame
    e2e: pd.DataFrame
    reuse_manifest: pd.DataFrame


# --------------------------------------------------------------------------- #
# Frozen target loading (reuse Study 1's LLM variants, no generation)
# --------------------------------------------------------------------------- #
def load_frozen_targets(
    queries: Sequence,
    points: Sequence,
    store: Optional[Dict[str, dict]] = None,
) -> Tuple[Dict[str, List[Article]], pd.DataFrame]:
    """Rebuild OFAT target shells and overlay Study 1's frozen LLM text.

    The article id = stable_id("art", query.id, profile, authenticity, suffix) is
    independent of the generation route, so a template-route shell has the same
    id as the frozen llm variant; we look that id up in the variant store and
    copy its text in via `apply_record`. Targets with no frozen `llm` record are
    kept as template shells but flagged in the manifest and warned about (never
    silently regenerated).
    """
    store = store if store is not None else load_variant_store()
    targets_by_query: Dict[str, List[Article]] = {}
    rows: List[dict] = []
    n_missing = 0
    for q in queries:
        shells = build_targets(q, points, route="template")
        for art in shells:
            rec = store.get(art.id)
            if rec is not None and rec.get("generator") == "llm":
                apply_record(art, rec)
                source = "reused"
            else:
                source = "template_fallback"
                n_missing += 1
            meta = art.meta or {}
            rows.append({
                "query_id": q.id,
                "domain": q.domain,
                "article_id": art.id,
                "target_dim": meta.get("target_dim", ""),
                "role": meta.get("role", ""),
                "n_chars": art.n_chars,
                "source": source,
            })
        targets_by_query[q.id] = shells
    if n_missing:
        warnings.warn(
            f"Study5: {n_missing} 个目标文在冻结变体库中缺少 llm 记录，已用模板壳子占位。"
            "\n请确认题目集与 Study 1 一致，或先运行 `gen-variants --query-source pool`。"
        )
    manifest = pd.DataFrame(rows)
    return targets_by_query, manifest


# --------------------------------------------------------------------------- #
# Corpus competitors as Article objects
# --------------------------------------------------------------------------- #
def _corpus_article(doc: CorpusDoc, query_id: str) -> Article:
    return Article(
        id=doc.doc_id,
        query_id=query_id,
        text=doc.text,
        is_target=False,
        meta={"role": "corpus", "generator": "real_passage", "src_query_id": doc.src_query_id},
    )


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def _add_profile_cols(row: dict, profile: Dict[str, int]) -> None:
    for dim in all_ids():
        row[dim] = profile.get(dim, 0)


def e2e_decomposition(
    retrieval_frame: pd.DataFrame, gen_frame: pd.DataFrame
) -> pd.DataFrame:
    """Per-factor retrieval / citation / end-to-end selection probabilities.

    Scoped to each factor's own OFAT pairs (target_dim == factor), comparing the
    treatment (top) vs control (baseline) level. End-to-end combines the two
    channels: P_e2e ~= P_retrieve * P_cite|ctx.
    """
    rows: List[dict] = []
    for f in all_ids():
        dim = get_dimension(f)
        lo, hi = dim.baseline_code(), dim.top_code()
        rsub = retrieval_frame[retrieval_frame.get("target_dim", "") == f]
        gsub = gen_frame[gen_frame.get("target_dim", "") == f]
        if rsub.empty or gsub.empty:
            continue

        def _m(df, col, level):
            s = df[df[f] == level][col]
            return float(s.mean()) if len(s) else float("nan")

        p_retr_c = _m(rsub, "retrieved", lo)
        p_retr_t = _m(rsub, "retrieved", hi)
        p_cite_c = _m(gsub, "y", lo)
        p_cite_t = _m(gsub, "y", hi)
        e2e_c = p_retr_c * p_cite_c
        e2e_t = p_retr_t * p_cite_t
        rows.append({
            "factor": f,
            "level_control": lo,
            "level_treated": hi,
            "P_retrieve_control": p_retr_c,
            "P_retrieve_treated": p_retr_t,
            "ATE_retrieve": p_retr_t - p_retr_c,
            "P_cite_ctx_control": p_cite_c,
            "P_cite_ctx_treated": p_cite_t,
            "ATE_cite": p_cite_t - p_cite_c,
            "P_e2e_control": e2e_c,
            "P_e2e_treated": e2e_t,
            "ATE_e2e": e2e_t - e2e_c,
        })
    return pd.DataFrame(rows)


def _scoped_ate(frame: pd.DataFrame, outcome: str, pair_key: str) -> pd.DataFrame:
    """ATE table where each factor is confined to its own OFAT pairs.

    Relies on `pair_key` containing the pair id (pair_<dim>), so within a group
    only the target dimension has both levels present (mirrors Study 1's scoping).
    """
    return ate_table(frame, outcome=outcome, paired_key=pair_key, cluster="query_id")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_study5(
    models: Sequence[str],
    per_domain: int = 1,
    domains: Optional[Sequence[str]] = None,
    prompt_styles: Sequence[str] = ("neutral",),
    seeds: Sequence[int] = (0,),
    top_k: int = RAG_TOP_K,
    retriever: str = RAG_RETRIEVER,
    alpha: float = RAG_HYBRID_ALPHA,
    query_source: str = "pool",
    mock: Optional[bool] = None,
    progress: bool = False,
    concurrency: int = 1,
    progress_file=None,
    output_sink: Optional["StudyModelSink"] = None,
    use_llm_cache: Optional[bool] = None,
    **_ignored,
) -> Study5Result:
    domains = list(domains or DOMAINS)
    queries = get_queries(query_source, per_domain=per_domain, domains=domains)
    points = ofat_pairs()

    # --- targets: reuse frozen Study 1 LLM variants (no generation) ---------
    targets_by_query, reuse_manifest = load_frozen_targets(queries, points)

    # --- build one retriever per domain (embeddings cached on disk) ----------
    retrievers: Dict[str, Optional[HybridRetriever]] = {}
    for dom in domains:
        docs = load_corpus(dom)
        if not docs:
            warnings.warn(
                f"domain {dom}: 语料库为空，跳过该领域。先运行 `python -m ai_structural_holes.cli build-corpus`。"
            )
            retrievers[dom] = None
        else:
            retrievers[dom] = HybridRetriever(docs, mode=retriever, alpha=alpha)

    # --- stage 1 (retrieval) + assemble stage-2 candidate sets --------------
    articles_by_id: Dict[str, Article] = {}
    queries_by_id = {q.id: q for q in queries}
    candidate_sets: List[CandidateSet] = []
    retrieval_rows: List[dict] = []

    total_targets = sum(len(targets_by_query.get(q.id, [])) for q in queries)
    retr_bar = make_progress_bar(total_targets, desc="Study5 检索", unit="目标", enabled=progress)

    for dom in domains:
        retr = retrievers.get(dom)
        dom_queries = [q for q in queries if q.domain == dom]
        dom_pairs = [
            (q, t) for q in dom_queries for t in targets_by_query.get(q.id, [])
        ]
        if retr is None:
            if retr_bar is not None:
                retr_bar.update(len(dom_pairs))
            continue

        # Batch all dense encoding for this domain in two model calls (queries +
        # targets) instead of one call per (query, target).
        target_emb_by_id: Dict[str, object] = {}
        if getattr(retr, "use_dense", False):
            retr.warm_queries([q.text for q in dom_queries])
            tgt_texts = [t.text for (_, t) in dom_pairs]
            if tgt_texts:
                embs = retr.embed(tgt_texts)
                target_emb_by_id = {t.id: e for (_, t), e in zip(dom_pairs, embs)}

        for q, tgt in dom_pairs:
            res = retr.retrieve(
                q.text, tgt.text, top_k, target_emb=target_emb_by_id.get(tgt.id)
            )
            articles_by_id[tgt.id] = tgt

            meta = tgt.meta or {}
            row = {
                "query_id": q.id,
                "domain": q.domain,
                "target_id": tgt.id,
                "target_dim": meta.get("target_dim", ""),
                "role": meta.get("role", ""),
                "pair_id": meta.get("pair_id", ""),
                "authenticity": tgt.authenticity,
                "retrieved": int(res.retrieved),
                "target_rank": res.target_rank,
                "target_score": res.target_score,
                "pair_key": f"{q.id}|{meta.get('pair_id', '')}",
            }
            _add_profile_cols(row, tgt.intended_profile)
            retrieval_rows.append(row)

            # stage-2 competitors: top (k-1) real corpus docs, forced-include target
            competitors = [
                _corpus_article(retr.doc_by_id[did], q.id)
                for did in res.competitor_doc_ids
            ]
            if not competitors:
                if retr_bar is not None:
                    retr_bar.update(1)
                continue
            for c in competitors:
                articles_by_id[c.id] = c
            set_size = len(competitors) + 1
            sets = build_candidate_sets(
                tgt, competitors, set_size=set_size,
                competitor_quality="real",
                counterbalance="all_positions",
                rng=random.Random(hash(tgt.id) & 0xFFFF),
                fixed_distractors=competitors,
            )
            candidate_sets.extend(sets)
            if retr_bar is not None:
                retr_bar.update(1)

    if retr_bar is not None:
        retr_bar.close()

    retrieval_frame = pd.DataFrame(retrieval_rows)
    if output_sink is not None:
        output_sink.set_retrieval_frame(retrieval_frame)

    # --- stage 2 (generation / citation) ------------------------------------
    runner = ExperimentRunner(
        models=models, prompt_styles=prompt_styles, seeds=seeds, mock=mock,
        concurrency=concurrency, output_mode="cite", use_llm_cache=use_llm_cache,
    )
    if output_sink is not None:
        output_sink.bind_context(articles_by_id, queries_by_id)
    if candidate_sets:
        gen_frame = runner.run(
            queries_by_id, articles_by_id, candidate_sets,
            progress=progress, desc="Study5 生成引用",
            progress_file=progress_file,
            on_trial=output_sink.on_trial if output_sink else None,
        )
    else:
        gen_frame = pd.DataFrame()

    # --- analysis ------------------------------------------------------------
    ate_retrieved = (
        _scoped_ate(retrieval_frame, outcome="retrieved", pair_key="pair_key")
        if not retrieval_frame.empty else pd.DataFrame()
    )

    ate_cite = pd.DataFrame()
    ei = pd.DataFrame()
    if not gen_frame.empty:
        gen_analysis = gen_frame
        if "parse_ok" in gen_frame.columns:
            gen_analysis = gen_frame[gen_frame["parse_ok"] == 1].copy()
        gen_analysis = gen_analysis.copy()
        gen_analysis["pair_key"] = (
            gen_analysis["query_id"].astype(str)
            + "|" + gen_analysis["model"].astype(str)
            + "|" + gen_analysis["prompt_style"].astype(str)
            + "|" + gen_analysis["target_position"].astype(str)
            + "|" + gen_analysis["seed"].astype(str)
            + "|" + gen_analysis.get("pair_id", "").astype(str)
        )
        ate_cite = ate_table(
            gen_analysis, outcome="y", paired_key="pair_key", cluster="query_id"
        )
        ei = ei_leverage_table(
            gen_analysis, route="experimental", scope_col="target_dim"
        )

    e2e = (
        e2e_decomposition(retrieval_frame, gen_frame)
        if not (retrieval_frame.empty or gen_frame.empty) else pd.DataFrame()
    )

    return Study5Result(
        retrieval_frame=retrieval_frame,
        gen_frame=gen_frame,
        ate_retrieved=ate_retrieved,
        ate_cite=ate_cite,
        ei=ei,
        e2e=e2e,
        reuse_manifest=reuse_manifest,
    )
