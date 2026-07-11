"""Core data structures: queries, candidate articles, candidate sets, trials.

A *FeatureProfile* maps dimension ids (S1..S4, O1..O4) to integer level codes
defined in the codebook. Articles carry their (intended) profile plus a
*verified* profile filled in by the manipulation check. Trials record one model
call over one ordered candidate set and the resulting decision Y per candidate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from ..codebook import all_ids, baseline_profile

FeatureProfile = Dict[str, int]


def normalize_profile(profile: Optional[FeatureProfile]) -> FeatureProfile:
    """Fill any missing dimensions with their baseline code (0)."""
    base = baseline_profile()
    if profile:
        base.update({k: int(v) for k, v in profile.items() if k in base})
    return base


def profile_key(profile: FeatureProfile) -> str:
    """Stable string key for a feature profile (for grouping/caching)."""
    p = normalize_profile(profile)
    return "|".join(f"{k}={p[k]}" for k in all_ids())


@dataclass
class Query:
    id: str
    domain: str
    text: str
    # The factual core / ground-truth answer this query revolves around. Kept
    # constant across all article variants so only S/O is manipulated.
    factual_core: str = ""


@dataclass
class Article:
    """A candidate article for one query."""

    id: str
    query_id: str
    text: str
    is_target: bool = False
    # 'genuine' vs 'fake' (Study 4): whether evidence/expertise is verifiable.
    authenticity: str = "genuine"
    intended_profile: FeatureProfile = field(default_factory=baseline_profile)
    verified_profile: Optional[FeatureProfile] = None
    manipulation_ok: Optional[bool] = None
    n_chars: int = 0
    # free-form design metadata (e.g. pair_id, design_label) copied into frames
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.intended_profile = normalize_profile(self.intended_profile)
        if not self.n_chars:
            self.n_chars = len(self.text)

    @property
    def profile(self) -> FeatureProfile:
        """Profile used in analysis: verified if available else intended."""
        return self.verified_profile or self.intended_profile


@dataclass
class CandidateSet:
    """An ordered list of candidate ids presented to a model (encodes R)."""

    query_id: str
    ordered_ids: List[str]  # presentation order (position = index)
    target_id: str
    competitor_quality: str = "mixed"  # strong | weak | mixed

    @property
    def size(self) -> int:
        return len(self.ordered_ids)

    def target_position(self) -> int:
        # -1 when the target is absent from the candidate set (e.g. Study 6's
        # real end-to-end RAG, where a non-retrieved target is not presented).
        try:
            return self.ordered_ids.index(self.target_id)
        except ValueError:
            return -1


@dataclass
class Trial:
    """One model decision over one ordered candidate set.

    `chosen_ids` is the model's selection; `y` maps candidate_id -> 0/1 selected.
    `scores` optionally holds per-candidate credibility scores or ranks.
    """

    trial_id: str
    query_id: str
    model: str
    prompt_style: str
    candidate_set: CandidateSet
    seed: int
    temperature: float
    chosen_ids: List[str] = field(default_factory=list)
    y: Dict[str, int] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    rank: Dict[str, int] = field(default_factory=dict)
    raw_response: str = ""
    parse_ok: bool = True
    # Non-empty when the API permanently rejected this request (e.g. a
    # content-filter block). Such trials carry no valid decision and are
    # excluded from analysis as missing data (not counted as y=0).
    api_error: str = ""

    def target_y(self) -> int:
        return int(self.y.get(self.candidate_set.target_id, 0))

    def to_row(self) -> dict:
        """Flatten to a tabular row for the target candidate (for analysis)."""
        cs = self.candidate_set
        return {
            "trial_id": self.trial_id,
            "query_id": self.query_id,
            "model": self.model,
            "prompt_style": self.prompt_style,
            "set_size": cs.size,
            "competitor_quality": cs.competitor_quality,
            "target_position": cs.target_position(),
            "seed": self.seed,
            "temperature": self.temperature,
            "y": self.target_y(),
            "target_score": self.scores.get(cs.target_id, float("nan")),
            "target_rank": self.rank.get(cs.target_id, -1),
            "parse_ok": int(self.parse_ok),
            "api_error": self.api_error,
        }


def stable_id(*parts: object) -> str:
    """Deterministic short id from arbitrary parts."""
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def dataclass_to_dict(obj) -> dict:
    return asdict(obj)
