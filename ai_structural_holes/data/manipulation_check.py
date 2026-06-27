"""Manipulation checks: verify the intended feature profile was realized.

Two layers:
  - rule-based detector (`detect_profile`): uses codebook presence-markers and
    simple heuristics to infer the realized level of each dimension from text.
    Fast, deterministic, used to gate generated variants.
  - validity gates (`check_article`, `check_pair`): confirm (a) the target
    dimension is at its intended level and (b) only the target dimension changed
    relative to a baseline, plus length control (+-10%).

For production use, the rule-based detector can be swapped for an independent
classifier or human coder; the interface (text -> FeatureProfile) is the same.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..codebook import ALL_DIMENSIONS, all_ids, get_dimension
from .schema import Article, FeatureProfile, normalize_profile


def _count_markers(text: str, markers: List[str]) -> int:
    return sum(text.count(m) for m in markers)


def detect_profile(text: str) -> FeatureProfile:
    """Infer realized levels from text using codebook markers + heuristics."""
    prof: Dict[str, int] = {}

    # S1: 0 none / 1 vague / 2 solid (needs numeric/CI/source)
    has_number = any(ch.isdigit() for ch in text) and ("%" in text or "CI" in text or "N=" in text)
    has_named_source = any(m in text for m in ("机构", "来源", "评分", "et al"))
    if has_number and has_named_source:
        prof["S1"] = 2
    elif any(m in text for m in ("研究表明", "有研究", "大家", "据说")):
        prof["S1"] = 1
    else:
        prof["S1"] = 0

    prof["S2"] = 1 if _count_markers(text, get_dimension("S2").presence_markers) > 0 else 0
    prof["S3"] = 1 if _count_markers(text, ("机制", "框架", "范式")) > 0 else 0
    # key S4 on distinctive markers so structural scaffolding (## 结论 / 结论先行)
    # in O2/O3 does not get misread as a substantive claim.
    prof["S4"] = 1 if _count_markers(text, ("选择理由", "核心优势")) > 0 else 0

    # O1: discrete units
    prof["O1"] = 1 if (text.count("\n- ") + text.count("\n* ") + text.count("|") > 0 or text.strip().startswith("- ")) else 0
    # O2: conclusion-first
    prof["O2"] = 1 if "结论先行" in text or "摘要" in text or "TL;DR" in text else 0
    # O3: functional headings
    prof["O3"] = 1 if ("##" in text or "#" in text or "适用场景" in text) else 0
    # O4: 0 distant / 1 adjacent / 2 bound (structural markers, orthogonal to S1)
    if "(证据:" in text or "（证据:" in text:
        prof["O4"] = 2
    elif "紧邻依据" in text:
        prof["O4"] = 1
    else:
        prof["O4"] = 0

    return normalize_profile(prof)


def length_ok(text_a: str, text_b: str, tol: float = 0.10) -> bool:
    la, lb = len(text_a), len(text_b)
    if la == 0:
        return lb == 0
    return abs(la - lb) / la <= tol


def check_article(article: Article) -> Article:
    """Fill verified_profile + manipulation_ok by comparing intended vs detected."""
    detected = detect_profile(article.text)
    article.verified_profile = detected
    intended = normalize_profile(article.intended_profile)
    article.manipulation_ok = all(detected[k] == intended[k] for k in all_ids())
    return article


def check_pair(
    baseline: Article,
    variant: Article,
    target_dim: str,
    length_tol: float = 0.10,
    require_length: bool = True,
) -> Tuple[bool, Dict[str, object]]:
    """Verify variant changed ONLY `target_dim` relative to baseline.

    Returns (ok, report). Report lists any off-target dimensions that drifted and
    whether the length constraint held. `require_length` controls whether a
    length-drift fails the gate (set False for the offline template route, where
    adding/removing a feature inherently changes length; keep True for the
    length-locked LLM-edited route).
    """
    db = detect_profile(baseline.text)
    dv = detect_profile(variant.text)
    drifted = [k for k in all_ids() if k != target_dim and db[k] != dv[k]]
    target_changed = db[target_dim] != dv[target_dim]
    len_ok = length_ok(baseline.text, variant.text, length_tol)
    ok = target_changed and not drifted and (len_ok or not require_length)
    report = {
        "target_dim": target_dim,
        "target_changed": target_changed,
        "off_target_drift": drifted,
        "length_ok": len_ok,
        "require_length": require_length,
        "baseline_detected": db,
        "variant_detected": dv,
    }
    return ok, report


def batch_check(articles: List[Article]) -> Dict[str, float]:
    """Return manipulation-check pass-rate stats over a list of articles."""
    if not articles:
        return {"n": 0, "pass_rate": float("nan")}
    checked = [check_article(a) for a in articles]
    n_ok = sum(1 for a in checked if a.manipulation_ok)
    return {"n": len(checked), "n_ok": n_ok, "pass_rate": n_ok / len(checked)}
