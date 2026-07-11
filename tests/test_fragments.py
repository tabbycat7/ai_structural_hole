"""Tests: every frozen phrasing variant realizes exactly its own dimension,
and template generation is deterministic per variant_seed."""
import pytest

from ai_structural_holes.codebook import all_ids, baseline_profile, get_dimension
from ai_structural_holes.data.fragments import FAMILIES, OPENERS, O2_TAIL, choose
from ai_structural_holes.data.generation import build_article_text, make_article, make_queries
from ai_structural_holes.data.manipulation_check import check_pair, detect_profile

CORE = "测试主题"
OPENER = OPENERS[0].format(core=CORE)
TAIL = O2_TAIL[0]


def _embed(family: str, frag: str) -> str:
    """Place one fragment into a minimal article context for detection."""
    frag = frag.format(core=CORE)
    if family == "O2.lead":
        return frag + OPENER
    if family in ("O4.bound", "O4.adjacent"):
        return OPENER.rstrip("。") + frag + TAIL
    if family in ("opener", "O2.tail"):
        return frag
    # semantic fragments: opener + fragment + conclusion-last tail
    return OPENER + frag + TAIL


@pytest.mark.parametrize("family", list(FAMILIES.keys()))
def test_every_variant_triggers_exactly_its_dimension(family):
    options, expected = FAMILIES[family]
    for frag in options:
        text = _embed(family, frag)
        detected = detect_profile(text)
        want = {**baseline_profile(), **expected}
        mismatched = {k: (detected[k], want[k]) for k in all_ids() if detected[k] != want[k]}
        assert not mismatched, f"{family} 素材串味: {frag!r} -> {mismatched}"


def test_choose_is_deterministic_and_seed_sensitive():
    a = choose(OPENERS, "seed-x", "opener")
    b = choose(OPENERS, "seed-x", "opener")
    assert a == b
    assert choose(OPENERS, None, "opener") == OPENERS[0]
    picks = {choose(OPENERS, f"seed-{i}", "opener") for i in range(30)}
    assert len(picks) > 1  # different seeds actually vary the phrasing


def test_build_article_text_deterministic():
    prof = {**baseline_profile(), "S1": 2, "S2": 1, "O2": 1}
    t1 = build_article_text(prof, CORE, variant_seed="q1")
    t2 = build_article_text(prof, CORE, variant_seed="q1")
    assert t1 == t2
    texts = {build_article_text(prof, CORE, variant_seed=f"q{i}") for i in range(10)}
    assert len(texts) > 1


@pytest.mark.parametrize("seed_i", range(8))
@pytest.mark.parametrize("dim", ["S1", "S2", "S3", "S4", "O1", "O2", "O3"])
def test_pair_purity_holds_across_variant_seeds(dim, seed_i):
    """OFAT purity must hold no matter which phrasing variant is drawn."""
    query = make_queries(per_domain=1)[0]
    seed = f"vseed-{seed_i}"
    base = make_article(query, baseline_profile(), variant_seed=seed)
    prof = baseline_profile()
    prof[dim] = get_dimension(dim).top_code()
    var = make_article(query, prof, variant_seed=seed)
    ok, rep = check_pair(base, var, dim, require_length=False)
    assert ok, rep["off_target_drift"]


@pytest.mark.parametrize("seed_i", range(8))
def test_o4_pair_purity_across_variant_seeds(seed_i):
    query = make_queries(per_domain=1)[0]
    seed = f"vseed-{seed_i}"
    s1_top = get_dimension("S1").top_code()
    base = make_article(query, {**baseline_profile(), "S1": s1_top}, variant_seed=seed)
    var = make_article(
        query, {**baseline_profile(), "S1": s1_top, "O4": 2}, variant_seed=seed
    )
    ok, rep = check_pair(base, var, "O4", require_length=False)
    assert ok, rep["off_target_drift"]


def test_fake_variants_carry_same_markers_as_genuine():
    """Study 4 relies on fake articles being marker-identical to genuine ones."""
    prof = {**baseline_profile(), "S1": 2, "S3": 1}
    genuine = build_article_text(prof, CORE, fake=False, variant_seed="q1")
    fake = build_article_text(prof, CORE, fake=True, variant_seed="q1")
    assert genuine != fake
    assert detect_profile(genuine) == detect_profile(fake)
