"""Shared helpers for building a study's articles, distractors, and sets.

Target generation route:
  - route="template" (default): frozen-fragment assembly, fully offline.
  - route="llm": every article is an LLM rewrite of the query's frozen base
    article (see data/base_articles.py), gated by the manipulation check.
    Queries without a frozen base article (or without a generation client)
    fall back to the template route with a warning.

Distractor route (independent of the target route):
  - None (default): same as the target route.
  - "real": frozen real web passages from the query pool (see
    data/query_pool.py); shortfalls are filled with template distractors.

Query source:
  - "builtin" (default): the hand-written topics in data/generation.py.
  - "pool": frozen real user questions imported via `cli import-queries`.
"""
from __future__ import annotations

import random
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

from ..codebook import baseline_profile, get_dimension
from ..data.generation import make_article, make_queries
from ..data.schema import Article, CandidateSet, Query
from ..task.protocol import build_candidate_sets
from .design import DesignPoint


def make_progress_bar(
    total: int,
    desc: str,
    unit: str = "篇",
    *,
    enabled: bool = True,
):
    """Return a tqdm bar, or None if disabled / unavailable."""
    if not enabled or total <= 0:
        return None
    try:
        from tqdm import tqdm

        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    except Exception:
        return None


def _article_gen_postfix(article: Article, query: Query) -> str:
    gen = (article.meta or {}).get("generator", "?")
    return f"{query.domain} [{gen}]"


def _count_llm_article_work(
    n_queries: int,
    *,
    route: str,
    gen_client,
    n_points: int = 0,
    n_distractors: int = 0,
    distractor_route: Optional[str] = None,
) -> int:
    """Estimate how many LLM article generations assemble will perform."""
    if not gen_client:
        return 0
    total = 0
    if route == "llm":
        total += n_queries * n_points
    d_route = distractor_route or route
    if d_route == "llm":
        total += n_queries * n_distractors
    return total


def get_queries(
    query_source: str = "builtin",
    per_domain: int = 1,
    domains: Optional[Sequence[str]] = None,
) -> List[Query]:
    """Resolve the query list from the chosen source (pool falls back)."""
    if query_source == "pool":
        from ..data.query_pool import load_pool_queries

        queries = load_pool_queries(per_domain=per_domain, domains=domains)
        if queries:
            return queries
        warnings.warn(
            "冻结题库为空，回退内置题目(先运行 `python -m ai_structural_holes.cli import-queries`)"
        )
    return make_queries(per_domain=per_domain, domains=domains)


def resolve_route(
    query: Query,
    route: str,
    client,
    base_texts: Dict[str, str],
) -> Tuple[str, Optional[str]]:
    """Pick the effective route + base text for one query (with fallback)."""
    if route != "llm":
        return "template", None
    base_text = base_texts.get(query.id)
    if client is None or not base_text:
        warnings.warn(
            f"query {query.id}: 缺少冻结基线文章或生成客户端，回退模板路线 "
            "(先运行 `python -m ai_structural_holes.cli gen-base`)"
        )
        return "template", None
    return "llm", base_text


def make_distractor_pool(
    query: Query,
    n: int = 4,
    rng: Optional[random.Random] = None,
    *,
    route: str = "template",
    client=None,
    gen_model: Optional[str] = None,
    base_text: Optional[str] = None,
    passages: Optional[List[dict]] = None,
    progress_bar=None,
    defer: Optional[list] = None,
) -> List[Article]:
    """Build competitor articles.

    route="real": wrap frozen real web passages (drawn without replacement);
    a shortfall is topped up with template distractors and warned about.
    Otherwise: generate mixed, moderate feature profiles via `make_article`.
    """
    rng = rng or random.Random(hash(query.id) & 0xFFFF)
    pool: List[Article] = []

    gen_route = route
    if route == "real":
        from ..data.query_pool import passage_to_article

        available = list(passages or [])
        chosen = rng.sample(available, min(n, len(available)))
        for i, p in enumerate(chosen):
            pool.append(passage_to_article(query, p, i))
        if len(pool) < n:
            warnings.warn(
                f"query {query.id}: 真实段落不足({len(pool)}/{n})，缺口回退模板陪跑"
            )
        gen_route = "template"  # top-up route for any shortfall

    for i in range(len(pool), n):
        prof = baseline_profile()
        # randomly switch on a couple of binary dims to create varied competitors
        for dim in ("S2", "S3", "S4", "O1", "O3"):
            if rng.random() < 0.4:
                prof[dim] = get_dimension(dim).top_code()
        pool.append(
            make_article(
                query, prof, is_target=False, suffix=f"distract{i}",
                meta={"role": "distractor"},
                # distinct phrasing per distractor so competitors aren't clones
                variant_seed=f"{query.id}|distract{i}",
                route=gen_route, client=client, gen_model=gen_model, base_text=base_text,
                defer=defer,
            )
        )
        if progress_bar is not None and gen_route == "llm" and defer is None:
            progress_bar.set_postfix_str(_article_gen_postfix(pool[-1], query))
            progress_bar.update(1)
    return pool


def build_targets(
    query: Query,
    points: Sequence[DesignPoint],
    authenticity: str = "genuine",
    *,
    route: str = "template",
    client=None,
    gen_model: Optional[str] = None,
    base_text: Optional[str] = None,
    progress_bar=None,
    defer: Optional[list] = None,
) -> List[Article]:
    """Materialize one target article per design point, carrying design meta."""
    arts = []
    for idx, p in enumerate(points):
        meta = {
            "design_label": p.label,
            "role": p.role,
            "pair_id": f"{query.id}|{p.label}" if p.label else "",
        }
        if p.target_dim:
            meta["target_dim"] = p.target_dim
        art = make_article(
            query, p.profile, is_target=True, authenticity=authenticity,
            suffix=f"{p.label}-{p.role}-{idx}", meta=meta,
            route=route, client=client, gen_model=gen_model, base_text=base_text,
            defer=defer,
        )
        arts.append(art)
        if progress_bar is not None and route == "llm" and defer is None:
            progress_bar.set_postfix_str(_article_gen_postfix(art, query))
            progress_bar.update(1)
    return arts


def run_deferred_jobs(
    jobs: list,
    queries_by_id: Dict[str, Query],
    *,
    concurrency: int = 1,
    progress: bool = False,
    use_variant_store: bool = True,
) -> None:
    """Execute queued deferred llm article jobs (optionally concurrently).

    RNG and article ids were already fixed during the sequential phase, so the
    execution order here does not affect reproducibility; results are filled into
    each job's article in place.

    When `use_variant_store` is set, each job first tries the frozen variant
    store (`data/variant_articles/`): a matching record (same article id, rewrite
    model, and base-text hash) is reused without any API call, and every freshly
    generated `llm` variant is written back for future runs.
    """
    if not jobs:
        return
    from ..data.generation import run_llm_job
    from ..llm.parallel import map_concurrent

    store: Dict[str, dict] = {}
    if use_variant_store:
        from ..data.variant_articles import apply_record, is_hit, load_variant_store

        store = load_variant_store()

    gen_bar = make_progress_bar(len(jobs), desc="生成变体文章", enabled=progress)

    def _advance(art, tag: str = "") -> None:
        if gen_bar is None:
            return
        q = queries_by_id.get(art.query_id)
        postfix = _article_gen_postfix(art, q) if q is not None else ""
        gen_bar.set_postfix_str(f"{postfix} {tag}".strip())
        gen_bar.update(1)

    pending: list = []
    if use_variant_store:
        for job in jobs:
            art = job["article"]
            rec = store.get(art.id)
            if rec is not None and is_hit(rec, gen_model=job["model"], base_text=job["base_text"]):
                apply_record(art, rec)
                _advance(art, "[store]")
            else:
                pending.append(job)
    else:
        pending = list(jobs)

    new_records: list = []

    def _on(_i, job, art):
        if use_variant_store and (art.meta or {}).get("generator") == "llm":
            from ..data.variant_articles import record_from_article

            new_records.append(record_from_article(art, job["base_text"], job["model"]))
        _advance(art)

    if pending:
        map_concurrent(run_llm_job, pending, concurrency=concurrency, on_result=_on)

    if use_variant_store and new_records:
        from ..data.variant_articles import save_variant_records

        save_variant_records(new_records)

    if gen_bar is not None:
        gen_bar.close()


def assemble(
    queries: Sequence[Query],
    points: Sequence[DesignPoint],
    set_size: int = 3,
    n_distractors: int = 4,
    counterbalance: str = "all_positions",
    competitor_quality: str = "mixed",
    authenticity: str = "genuine",
    seed: int = 0,
    route: str = "template",
    gen_client=None,
    gen_model: Optional[str] = None,
    distractor_route: Optional[str] = None,
    progress: bool = False,
    concurrency: int = 1,
    use_variant_store: bool = True,
) -> Tuple[Dict[str, Query], Dict[str, Article], List[CandidateSet]]:
    """Build the full (queries, articles, candidate sets) for a study.

    Each (query x design point) target gets position-counterbalanced sets drawn
    from that query's distractor pool. `distractor_route=None` means distractors
    follow the target route; "real" uses frozen query-pool passages.

    LLM article generation is two-phase: a sequential phase fixes all RNG draws /
    article ids and queues the LLM edits, then those edits run (up to
    `concurrency` at a time) without affecting reproducibility.
    """
    rng = random.Random(seed)
    queries_by_id = {q.id: q for q in queries}
    articles_by_id: Dict[str, Article] = {}
    candidate_sets: List[CandidateSet] = []

    base_texts: Dict[str, str] = {}
    if route == "llm":
        from ..data.base_articles import load_base_texts

        base_texts = load_base_texts()

    pool_passages: Dict[str, List[dict]] = {}
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
            q, n=n_distractors, rng=rng,
            route=d_route, client=gen_client, gen_model=gen_model, base_text=base_text,
            passages=pool_passages.get(q.id),
            defer=jobs,
        )
        for d in distractors:
            articles_by_id[d.id] = d
        # Draw the competitors ONCE per query and reuse them for every target of
        # this query (all dimensions, both control and treatment). This keeps the
        # competition environment identical across a matched pair, so the paired
        # contrast reflects only the target's own toggled feature, not a change
        # in who it competes against.
        shared_distractors = rng.sample(list(distractors), set_size - 1)
        targets = build_targets(
            q, points, authenticity=authenticity,
            route=q_route, client=gen_client, gen_model=gen_model, base_text=base_text,
            defer=jobs,
        )
        for tgt in targets:
            articles_by_id[tgt.id] = tgt
            sets = build_candidate_sets(
                tgt, distractors, set_size=set_size,
                competitor_quality=competitor_quality,
                counterbalance=counterbalance, rng=rng,
                fixed_distractors=shared_distractors,
            )
            candidate_sets.extend(sets)

    run_deferred_jobs(
        jobs, queries_by_id, concurrency=concurrency, progress=progress,
        use_variant_store=use_variant_store,
    )

    return queries_by_id, articles_by_id, candidate_sets


def make_gen_client(
    gen_route: str,
    mock: Optional[bool],
    dry_run: bool = False,
):
    """Client used for LLM-assisted article generation (None on template route).

    Dry runs never generate articles via API, so they always return None (the
    call-count estimate is route-independent anyway).
    """
    if gen_route != "llm" or dry_run:
        return None
    from ..llm.client import get_client

    return get_client(mock=mock, use_llm_cache=True)
