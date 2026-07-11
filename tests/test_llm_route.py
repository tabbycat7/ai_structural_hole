"""Tests: LLM-edited route (no rule-based gating; empty-response fallback) and
frozen base-article generation/loading."""
from types import SimpleNamespace

from ai_structural_holes.codebook import baseline_profile
from ai_structural_holes.data.base_articles import (
    generate_base_article,
    load_base_texts,
    save_base_article,
)
from ai_structural_holes.data.fragments import S2_PRESENT
from ai_structural_holes.data.generation import llm_edit_variant, make_article, make_queries

BASE_TEXT = "这里对「测试主题」做一个整体的介绍。" + "内容平实，叙述普通。" * 60
GOOD_S2_TEXT = BASE_TEXT + S2_PRESENT[0]  # adds only the S2 marker, within ±10% length


class StubClient:
    """Returns scripted responses in order; repeats the last one when exhausted."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.seen_messages = []

    def call(self, *, model, messages, temperature=0.0, seed=None, max_tokens=800):
        self.seen_messages.append(messages)
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return SimpleNamespace(text=self.responses[idx], cached=False)


def test_llm_edit_variant_accepts_first_nonempty_output():
    # No rule-based gating: the first non-empty rewrite is accepted verbatim,
    # even if it wouldn't have passed the old hardcoded marker check.
    client = StubClient(["模型改写后的一段正文，不含任何硬编码标记。", GOOD_S2_TEXT])
    target = {**baseline_profile(), "S2": 1}
    text, report = llm_edit_variant(
        client, base_text=BASE_TEXT, target_profile=target,
        core="测试主题", model="stub/model",
    )
    assert text == "模型改写后的一段正文，不含任何硬编码标记。"
    assert report["attempts"] == 1
    assert client.calls == 1


def test_llm_edit_variant_gives_up_only_on_empty():
    client = StubClient([""])  # empty every time
    text, report = llm_edit_variant(
        client, base_text=BASE_TEXT, target_profile={**baseline_profile(), "S2": 1},
        core="测试主题", model="stub/model", max_attempts=3,
    )
    assert text is None
    assert report["attempts"] == 3
    assert client.calls == 3


def test_make_article_llm_route_accepts_output_without_verification():
    query = make_queries(per_domain=1)[0]
    client = StubClient(["随便一段被直接接受的改写正文。"])
    art = make_article(
        query, {**baseline_profile(), "S2": 1},
        route="llm", client=client, gen_model="stub/model", base_text=BASE_TEXT,
    )
    assert art.text == "随便一段被直接接受的改写正文。"
    assert art.meta["generator"] == "llm"
    # verification was removed: analysis keys on the intended profile instead
    assert art.verified_profile is None
    assert art.manipulation_ok is None


def test_make_article_llm_route_falls_back_only_on_empty():
    query = make_queries(per_domain=1)[0]
    client = StubClient([""])  # never returns usable text
    art = make_article(
        query, {**baseline_profile(), "S2": 1},
        route="llm", client=client, gen_model="stub/model", base_text=BASE_TEXT,
    )
    assert art.meta["generator"] == "template_fallback"
    assert art.text  # template text produced


VALID_BASE = "这个主题在日常生活中比较常见，人们的做法各有不同，情况也随场合而变化。" * 4


def test_generate_base_article_validates_and_freezes(tmp_path):
    query = make_queries(per_domain=1)[0]
    client = StubClient([VALID_BASE])
    rec = generate_base_article(client, query, "stub/model")
    assert rec["generator"] == "llm"
    assert rec["validated"] is True
    assert all(v == 0 for v in rec["detected_profile"].values())

    save_base_article(rec, root=tmp_path)
    texts = load_base_texts(root=tmp_path)
    assert texts[query.id] == VALID_BASE.strip()


def test_generate_base_article_falls_back_on_bad_output(tmp_path):
    query = make_queries(per_domain=1)[0]
    client = StubClient(['{"choice": "A"}'])  # mock-style JSON, never valid prose
    rec = generate_base_article(client, query, "stub/model", max_attempts=2)
    assert rec["generator"] == "template_fallback"
    assert rec["validated"] is True  # template baseline is all-zero by construction
    save_base_article(rec, root=tmp_path)
    assert query.id in load_base_texts(root=tmp_path)


def _llm_variant(query, target):
    """A finalized llm variant Article (generator=='llm')."""
    client = StubClient([GOOD_S2_TEXT])
    return make_article(
        query, target,
        route="llm", client=client, gen_model="stub/model", base_text=BASE_TEXT,
    )


def test_variant_store_roundtrip_and_hit(tmp_path):
    from ai_structural_holes.data import variant_articles as vs

    query = make_queries(per_domain=1)[0]
    target = {**baseline_profile(), "S2": 1}
    art = _llm_variant(query, target)

    rec = vs.record_from_article(art, BASE_TEXT, "stub/model")
    vs.save_variant_records([rec], root=tmp_path)

    store = vs.load_variant_store(root=tmp_path)
    assert art.id in store
    loaded = store[art.id]
    assert vs.is_hit(loaded, gen_model="stub/model", base_text=BASE_TEXT)

    # applying the record onto a fresh (placeholder) article reproduces the variant
    fresh = make_article(
        query, target,
        route="llm", client=StubClient([GOOD_S2_TEXT]),
        gen_model="stub/model", base_text=BASE_TEXT, defer=[],
    )
    assert fresh.id == art.id
    vs.apply_record(fresh, loaded)
    assert fresh.text == art.text
    assert fresh.meta["generator"] == "llm"
    # verification was dropped; llm variants carry no verified profile
    assert fresh.verified_profile is None


def test_variant_store_invalidates_on_model_or_base_change(tmp_path):
    from ai_structural_holes.data import variant_articles as vs

    query = make_queries(per_domain=1)[0]
    art = _llm_variant(query, {**baseline_profile(), "S2": 1})
    rec = vs.record_from_article(art, BASE_TEXT, "stub/model")

    assert not vs.is_hit(rec, gen_model="other/model", base_text=BASE_TEXT)
    assert not vs.is_hit(rec, gen_model="stub/model", base_text=BASE_TEXT + "改动")


def test_variant_store_only_stores_successful_llm(tmp_path):
    from ai_structural_holes.data import variant_articles as vs

    query = make_queries(per_domain=1)[0]
    # a template_fallback article (only produced now when the model returns
    # empty) should be excluded by callers, and the record itself is never a hit
    # because generator != "llm"
    client = StubClient([""])
    art = make_article(
        query, {**baseline_profile(), "S2": 1},
        route="llm", client=client, gen_model="stub/model", base_text=BASE_TEXT,
    )
    assert art.meta["generator"] == "template_fallback"
    rec = vs.record_from_article(art, BASE_TEXT, "stub/model")
    assert not vs.is_hit(rec, gen_model="stub/model", base_text=BASE_TEXT)
