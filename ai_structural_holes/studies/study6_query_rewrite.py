"""Study 6 - query-rewrite-driven RAG retrieval.

Where Study 5 uses a *fixed*, model-independent retriever, Study 6 lets the
model under test participate in retrieval: before any document is seen, the
model rewrites/expands the user question into one or more search queries, and
those queries drive the same frozen `HybridRetriever`. Because rewriting sees
only the question (never the injected target), a query's control/treatment
targets still face an identical retrieval environment *under the same model*,
so retrieval now varies by model.

Study 6 is a *real* end-to-end RAG (unlike Study 5's two-channel decomposition):
  - No forced target inclusion: the candidate set is exactly the top-k that the
    model's rewritten queries actually retrieved, in natural score order. The
    target appears only when it was genuinely retrieved (`target_rank < top_k`),
    at its true rank; otherwise it is absent and cannot be cited (y=0).
  - No position counterbalancing: each target yields a single candidate set, so
    the generation stage costs 1 call per target (not top_k).
  - The outcome `y` is therefore the *end-to-end* probability that the target is
    finally cited (retrieval + citation combined), not a conditional cite rate.

Design reuse (shared with Study 5):
  - Target articles are the frozen Study 1 LLM variants (`load_frozen_targets`),
    so there is zero article generation cost.
  - The generation stage reuses `ExperimentRunner` with `output_mode="cite"`.
  - Competitors are real corpus passages.

Key difference from Study 5:
  - The retrieval channel (`retrieved`, `target_rank`) is now MODEL-DEPENDENT,
    so `retrieval.csv` / `ate_retrieved.csv` move into per-model outputs and a
    cross-model comparison (`retrieval_by_model.csv`) becomes the headline.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..study_output import StudyModelSink

from ..config import DOMAINS, RAG_HYBRID_ALPHA, RAG_RETRIEVER, RAG_TOP_K
from ..data.schema import Article, CandidateSet
from ..experiment.runner import ExperimentRunner
from ..llm.client import get_client
from ..llm.parallel import map_concurrent
from ..retrieval.corpus import load_corpus
from ..retrieval.retriever import HybridRetriever
from ..task.protocol import parse_rewrite
from ..task.prompts import build_rewrite_messages, max_tokens_for_rewrite
from .common import get_queries, make_progress_bar
from .design import ofat_pairs
# Reuse Study 5's frozen-target loader, corpus-article wrapper, profile columns,
# and scoped ATE (identical semantics).
from .study5_rag import (
    _add_profile_cols,
    _corpus_article,
    _scoped_ate,
    load_frozen_targets,
)


@dataclass
class Study6Result:
    retrieval_frame: pd.DataFrame  # per (model, target); retrieval is model-dependent
    gen_frame: pd.DataFrame        # citation trials, all models
    rewrite_frame: pd.DataFrame    # audit: what each model searched for
    reuse_manifest: pd.DataFrame


# Columns that uniquely identify a generation trial for resume matching. They
# are all present in trials.csv, so a resumed run can recognise completed work
# regardless of the trial-id formula (which cannot be recomputed from disk since
# ordered_ids is not persisted). (query_id, target_dim, role, pair_id) pins the
# exact target; (model, prompt_style, seed) pins the decision context.
_RESUME_KEY_COLS = (
    "query_id", "model", "prompt_style", "seed", "target_dim", "role", "pair_id",
)


def _make_resume_skip(done_keys: set, articles_by_id: Dict[str, Article]):
    """Predicate: has this (candidate set, model, style, seed) already run?"""
    def _skip(cs, model, style, seed) -> bool:
        meta = (articles_by_id.get(cs.target_id).meta or {}) \
            if cs.target_id in articles_by_id else {}
        key = (
            str(cs.query_id), str(model), str(style), str(seed),
            str(meta.get("target_dim", "")), str(meta.get("role", "")),
            str(meta.get("pair_id", "")),
        )
        return key in done_keys
    return _skip


def _natural_topk_ids(res, target_id: str, k: int) -> List[str]:
    """Real top-k of [corpus + target] in natural score order (no forced insert).

    The target is included only if it was actually retrieved (`target_rank < k`),
    at its true rank among the corpus docs; otherwise the top-k is pure corpus and
    the target is absent (so it cannot be cited -> end-to-end y=0).
    """
    corpus = list(res.top_k_doc_ids)  # already sorted by descending score
    if res.target_rank < k:
        return corpus[: res.target_rank] + [target_id] + corpus[res.target_rank : k - 1]
    return corpus[:k]


# --------------------------------------------------------------------------- #
# Stage 0: query rewriting (per model x query, target never seen)
# --------------------------------------------------------------------------- #
def _rewrite_all(
    models: Sequence[str],
    queries: Sequence,
    n_queries: int,
    client,
    concurrency: int,
    progress: bool,
    on_rewrite: Optional[Callable[[dict], None]] = None,
    preloaded: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> tuple[Dict[tuple, List[str]], pd.DataFrame]:
    """Rewrite each (model, query) into search queries; return map + audit frame.

    `preloaded[model][query_id]` supplies rewrites reused from a prior run
    (resume): those (model, query) pairs are not re-sent to the model and are
    not re-emitted via `on_rewrite` (they are already persisted). Only missing
    pairs incur an API call.
    """
    preloaded = preloaded or {}
    rewrites: Dict[tuple, List[str]] = {}
    reused_rows: List[dict] = []
    jobs = []
    for model in models:
        for q in queries:
            cached = preloaded.get(model, {}).get(q.id)
            if cached:
                rewrites[(model, q.id)] = cached
                reused_rows.append({
                    "model": model,
                    "query_id": q.id,
                    "domain": q.domain,
                    "original_query": q.text,
                    "rewritten_queries": " ||| ".join(cached),
                    "n_queries": len(cached),
                    "parse_ok": 1,
                })
            else:
                jobs.append((model, q))

    bar = make_progress_bar(len(jobs), desc="Study6 查询改写", unit="改写", enabled=progress)

    def _do(job):
        model, q = job
        messages = build_rewrite_messages(q.text, q.domain, n_queries)
        resp = client.call(
            model=model, messages=messages, temperature=0.0,
            max_tokens=max_tokens_for_rewrite(),
        )
        return parse_rewrite(getattr(resp, "text", ""), fallback_query=q.text)

    def _on(_i, _job, _res):
        if bar is not None:
            bar.update(1)

    results = map_concurrent(_do, jobs, concurrency=concurrency, on_result=_on)
    if bar is not None:
        bar.close()

    rows: List[dict] = list(reused_rows)
    for (model, q), parsed in zip(jobs, results):
        queries_list = parsed["queries"]
        rewrites[(model, q.id)] = queries_list
        row = {
            "model": model,
            "query_id": q.id,
            "domain": q.domain,
            "original_query": q.text,
            "rewritten_queries": " ||| ".join(queries_list),
            "n_queries": len(queries_list),
            "parse_ok": int(parsed["parse_ok"]),
        }
        rows.append(row)
        if on_rewrite is not None:
            on_rewrite(row)
    return rewrites, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_study6(
    models: Sequence[str],
    per_domain: int = 1,
    domains: Optional[Sequence[str]] = None,
    prompt_styles: Sequence[str] = ("neutral",),
    seeds: Sequence[int] = (0,),
    top_k: int = RAG_TOP_K,
    retriever: str = RAG_RETRIEVER,
    alpha: float = RAG_HYBRID_ALPHA,
    n_queries: int = 3,
    fuse: str = "max",
    query_source: str = "pool",
    mock: Optional[bool] = None,
    progress: bool = False,
    concurrency: int = 1,
    progress_file=None,
    output_sink: Optional["StudyModelSink"] = None,
    use_llm_cache: Optional[bool] = None,
    resume: bool = False,
    **_ignored,
) -> Study6Result:
    domains = list(domains or DOMAINS)
    models = list(models)
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

    # --- stage 0: query rewriting (model-dependent, target-independent) ------
    client = get_client(mock=mock, use_llm_cache=use_llm_cache)
    preloaded_rewrites = None
    if resume and output_sink is not None:
        preloaded_rewrites = {m: output_sink.load_rewrites(m) for m in models}
    rewrites, rewrite_frame = _rewrite_all(
        models,
        queries,
        n_queries,
        client,
        concurrency,
        progress,
        on_rewrite=output_sink.on_rewrite_row if output_sink else None,
        preloaded=preloaded_rewrites,
    )

    # --- stage 1 (retrieval) + assemble stage-2 candidate sets, per model ----
    articles_by_id: Dict[str, Article] = {}
    queries_by_id = {q.id: q for q in queries}
    candidate_sets_by_model: Dict[str, List[CandidateSet]] = {m: [] for m in models}
    retrieval_rows: List[dict] = []

    total = len(models) * sum(len(targets_by_query.get(q.id, [])) for q in queries)
    retr_bar = make_progress_bar(total, desc="Study6 检索", unit="目标", enabled=progress)

    for model in models:
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

            # Batch dense encoding for this (model, domain): all rewritten queries
            # + all target texts in two model calls instead of one per pair.
            target_emb_by_id: Dict[str, object] = {}
            if getattr(retr, "use_dense", False):
                all_rw: List[str] = []
                for q in dom_queries:
                    all_rw.extend(rewrites.get((model, q.id), [q.text]))
                retr.warm_queries(all_rw)
                tgt_texts = [t.text for (_, t) in dom_pairs]
                if tgt_texts:
                    embs = retr.embed(tgt_texts)
                    target_emb_by_id = {t.id: e for (_, t), e in zip(dom_pairs, embs)}

            for q, tgt in dom_pairs:
                rw = rewrites.get((model, q.id), [q.text])
                res = retr.retrieve_multi(
                    rw, tgt.text, top_k,
                    target_emb=target_emb_by_id.get(tgt.id), fuse=fuse,
                )
                articles_by_id[tgt.id] = tgt

                meta = tgt.meta or {}
                row = {
                    "model": model,
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

                # Real end-to-end candidate set: exactly the top-k the model's
                # rewritten queries retrieved, in natural score order. The target
                # is present only if it was genuinely retrieved (at its true rank);
                # otherwise it is absent and cannot be cited (y=0). No forced
                # inclusion, no position counterbalancing -> one set per target.
                for did in res.top_k_doc_ids:
                    articles_by_id[did] = _corpus_article(retr.doc_by_id[did], q.id)
                ordered_ids = _natural_topk_ids(res, tgt.id, top_k)
                if len(ordered_ids) < 2:
                    if retr_bar is not None:
                        retr_bar.update(1)
                    continue
                cs = CandidateSet(
                    query_id=q.id,
                    ordered_ids=ordered_ids,
                    target_id=tgt.id,
                    competitor_quality="real",
                )
                candidate_sets_by_model[model].append(cs)
                if retr_bar is not None:
                    retr_bar.update(1)

    if retr_bar is not None:
        retr_bar.close()

    retrieval_frame = pd.DataFrame(retrieval_rows)
    if output_sink is not None:
        output_sink.set_retrieval_frame(retrieval_frame, per_model=True)

    # --- stage 2 (generation / citation), one runner per model ---------------
    gen_frames: List[pd.DataFrame] = []
    for model in models:
        sets = candidate_sets_by_model.get(model, [])
        if not sets:
            continue
        runner = ExperimentRunner(
            models=[model], prompt_styles=prompt_styles, seeds=seeds, mock=mock,
            concurrency=concurrency, output_mode="cite", use_llm_cache=use_llm_cache,
        )
        if output_sink is not None:
            output_sink.bind_context(articles_by_id, queries_by_id)
        skip = None
        if resume and output_sink is not None:
            # Self-heal any duplicate rows a prior broken resume may have left,
            # then match completed trials on a semantic key (columns present in
            # trials.csv) so resume is robust to trial-id changes and re-runs.
            output_sink.dedupe_trials(model, _RESUME_KEY_COLS)
            done = output_sink.completed_row_keys(model, _RESUME_KEY_COLS)
            skip = _make_resume_skip(done, articles_by_id)
        gf = runner.run(
            queries_by_id, articles_by_id, sets,
            progress=progress, desc=f"Study6 生成引用[{model.split('/')[-1]}]",
            progress_file=progress_file,
            on_trial=output_sink.on_trial if output_sink else None,
            skip=skip,
        )
        if not gf.empty:
            gen_frames.append(gf)

    gen_frame = pd.concat(gen_frames, ignore_index=True) if gen_frames else pd.DataFrame()

    return Study6Result(
        retrieval_frame=retrieval_frame,
        gen_frame=gen_frame,
        rewrite_frame=rewrite_frame,
        reuse_manifest=reuse_manifest,
    )


# --------------------------------------------------------------------------- #
# Cross-model retrieval-channel summary (the headline of Study 6)
# --------------------------------------------------------------------------- #
def retrieval_by_model(retrieval_frame: pd.DataFrame) -> pd.DataFrame:
    """Per-model retrieval-channel factor ATEs + overall recall, stacked.

    The retrieval channel is model-dependent in Study 6, so comparing each
    model's per-factor ATE_retrieve (and overall recall) is the core question:
    does a structural-hole feature help retrieval more for some models than
    others, once the model chooses how to search?
    """
    if retrieval_frame.empty or "model" not in retrieval_frame.columns:
        return pd.DataFrame()
    rows: List[dict] = []
    for model, sub in retrieval_frame.groupby("model"):
        recall = float(sub["retrieved"].mean()) if len(sub) else float("nan")
        ate = _scoped_ate(sub, outcome="retrieved", pair_key="pair_key")
        for _, r in ate.iterrows():
            rows.append({
                "model": model,
                "factor": r.get("factor", ""),
                "ATE_retrieve": r.get("ATE", float("nan")),
                "ci_low": r.get("ci_low", float("nan")),
                "ci_high": r.get("ci_high", float("nan")),
                "recall_overall": recall,
            })
    return pd.DataFrame(rows)
