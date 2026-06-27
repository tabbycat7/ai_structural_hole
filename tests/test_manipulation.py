"""Tests: single-dimension manipulation changes only the target dimension."""
import pytest

from ai_structural_holes.codebook import baseline_profile, get_dimension
from ai_structural_holes.data.generation import make_article, make_queries
from ai_structural_holes.data.manipulation_check import check_pair, detect_profile


@pytest.fixture(scope="module")
def query():
    return make_queries(per_domain=1)[0]


@pytest.mark.parametrize("dim,level", [
    ("S1", 2), ("S2", 1), ("S3", 1), ("S4", 1), ("O1", 1), ("O2", 1), ("O3", 1),
])
def test_single_dim_no_off_target_drift(query, dim, level):
    base = make_article(query, baseline_profile())
    prof = baseline_profile()
    prof[dim] = level
    var = make_article(query, prof)
    ok, rep = check_pair(base, var, dim, require_length=False)
    assert ok, rep["off_target_drift"]


def test_o4_requires_s1(query):
    # O4 toggled on an S1-present base changes only O4.
    s1_top = get_dimension("S1").top_code()
    base = make_article(query, {**baseline_profile(), "S1": s1_top})
    var = make_article(query, {**baseline_profile(), "S1": s1_top, "O4": 2})
    ok, rep = check_pair(base, var, "O4", require_length=False)
    assert ok, rep["off_target_drift"]


def test_detect_baseline_all_zero(query):
    base = make_article(query, baseline_profile())
    det = detect_profile(base.text)
    assert all(v == 0 for v in det.values())
