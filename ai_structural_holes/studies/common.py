"""Shared helpers for building a study's articles, distractors, and sets."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..codebook import baseline_profile, get_dimension
from ..data.generation import make_article
from ..data.schema import Article, CandidateSet, Query
from ..task.protocol import build_candidate_sets
from .design import DesignPoint


def make_distractor_pool(query: Query, n: int = 4, rng: Optional[random.Random] = None) -> List[Article]:
    """Generate competitor articles with mixed, moderate feature profiles."""
    rng = rng or random.Random(hash(query.id) & 0xFFFF)
    pool: List[Article] = []
    for i in range(n):
        prof = baseline_profile()
        # randomly switch on a couple of binary dims to create varied competitors
        for dim in ("S2", "S3", "S4", "O1", "O3"):
            if rng.random() < 0.4:
                prof[dim] = get_dimension(dim).top_code()
        pool.append(
            make_article(query, prof, is_target=False, suffix=f"distract{i}",
                         meta={"role": "distractor"})
        )
    return pool


def build_targets(
    query: Query,
    points: Sequence[DesignPoint],
    authenticity: str = "genuine",
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
        arts.append(
            make_article(
                query, p.profile, is_target=True, authenticity=authenticity,
                suffix=f"{p.label}-{p.role}-{idx}", meta=meta,
            )
        )
    return arts


def assemble(
    queries: Sequence[Query],
    points: Sequence[DesignPoint],
    set_size: int = 3,
    n_distractors: int = 4,
    counterbalance: str = "all_positions",
    competitor_quality: str = "mixed",
    authenticity: str = "genuine",
    seed: int = 0,
) -> Tuple[Dict[str, Query], Dict[str, Article], List[CandidateSet]]:
    """Build the full (queries, articles, candidate sets) for a study.

    Each (query x design point) target gets position-counterbalanced sets drawn
    from that query's distractor pool.
    """
    rng = random.Random(seed)
    queries_by_id = {q.id: q for q in queries}
    articles_by_id: Dict[str, Article] = {}
    candidate_sets: List[CandidateSet] = []

    for q in queries:
        distractors = make_distractor_pool(q, n=n_distractors, rng=rng)
        for d in distractors:
            articles_by_id[d.id] = d
        targets = build_targets(q, points, authenticity=authenticity)
        for tgt in targets:
            articles_by_id[tgt.id] = tgt
            sets = build_candidate_sets(
                tgt, distractors, set_size=set_size,
                competitor_quality=competitor_quality,
                counterbalance=counterbalance, rng=rng,
            )
            candidate_sets.extend(sets)

    return queries_by_id, articles_by_id, candidate_sets
