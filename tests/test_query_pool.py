"""Tests: real-query pool import (DuReader format), loading, and the
real-passage distractor route with shortfall fallback."""
import json
import random

import pytest

from ai_structural_holes.data.query_pool import (
    MIN_PASSAGES,
    PASSAGE_CHARS,
    classify_domain,
    import_queries,
    load_pool_passages,
    load_pool_queries,
    passage_to_article,
    usable_passages,
)
from ai_structural_holes.data.schema import Query
from ai_structural_holes.studies.common import make_distractor_pool


def _passage(sentence: str, n: int = 8) -> str:
    """A cleanable, in-band passage built from a distinctive sentence."""
    return ("<p>" + sentence + "</p>") * n


_QUESTIONS = {
    "health": "成年人怎么补充维生素比较好",
    "finance": "指数基金定投应该怎么操作",
    "travel": "去云南旅游的行程怎么安排",
    "consumer_product": "千元价位的耳机哪款值得买",
    "academic_qa": "神经网络的原理是什么",
}


def _dureader_line(question: str, tag: str) -> str:
    docs = [
        {"paragraphs": [_passage(f"{tag}相关的第{i}篇网页内容，介绍了一些常见的做法与注意点。")]}
        for i in range(MIN_PASSAGES + 1)
    ]
    return json.dumps({"question": question, "documents": docs}, ensure_ascii=False)


@pytest.fixture()
def dureader_file(tmp_path):
    lines = [_dureader_line(q, dom) for dom, q in _QUESTIONS.items()]
    # noise: unclassifiable question + too-short question + too-few passages
    lines.append(_dureader_line("今天天气怎么样啊各位", "无领域"))
    lines.append(_dureader_line("短", "太短"))
    lines.append(json.dumps({
        "question": "血压高吃什么比较合适呢",
        "documents": [{"paragraphs": [_passage("唯一一篇。")]}],
    }, ensure_ascii=False))
    path = tmp_path / "dureader_dev.json"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_classify_domain():
    for dom, q in _QUESTIONS.items():
        assert classify_domain(q) == dom
    assert classify_domain("今天天气怎么样") is None
    # ambiguous (hits two domains equally) -> dropped
    assert classify_domain("旅游时买什么耳机") is None


def test_usable_passages_cleans_dedups_and_bands():
    lo, hi = PASSAGE_CHARS
    good = _passage("一段足够长的正常内容，用来测试清洗流程。")
    dup = good
    short = "<p>太短。</p>"
    long_ = "长句子内容。" * 200
    out = usable_passages([good, dup, short, long_])
    assert len(out) == 2  # dup and short dropped, long truncated + kept
    for p in out:
        assert lo <= len(p) <= hi
        assert "<p>" not in p


def test_import_and_load_roundtrip(dureader_file, tmp_path):
    root = tmp_path / "frozen"
    stats = import_queries(dureader_file, per_domain=10, seed=0, root=root)
    for dom in _QUESTIONS:
        assert stats[dom]["imported"] == 1, stats

    queries = load_pool_queries(root=root)
    assert len(queries) == 5
    assert {q.domain for q in queries} == set(_QUESTIONS)
    q = next(x for x in queries if x.domain == "health")
    assert q.text == _QUESTIONS["health"]
    assert q.factual_core == _QUESTIONS["health"]

    passages = load_pool_passages(root=root)
    assert set(passages) == {q.id for q in queries}
    for plist in passages.values():
        assert len(plist) >= MIN_PASSAGES
        for p in plist:
            assert "detected_profile" in p and p["n_chars"] == len(p["text"])

    # per_domain slicing
    assert len(load_pool_queries(per_domain=0, root=root)) == 0


def test_passage_to_article_carries_detected_profile():
    q = Query(id="q1", domain="health", text="问题", factual_core="问题")
    p = {"text": "一段真实网页内容。" * 20, "detected_profile": None}
    art = passage_to_article(q, p, 0)
    assert art.meta["generator"] == "real_passage"
    assert not art.is_target
    assert art.verified_profile is not None  # recomputed when not stored


def test_distractor_route_real_with_shortfall_fallback():
    q = Query(id="q1", domain="health", text="问题", factual_core="问题")
    passages = [
        {"text": f"第{i}段真实网页内容，讲了一些日常经验。" * 10} for i in range(2)
    ]
    with pytest.warns(UserWarning, match="真实段落不足"):
        pool = make_distractor_pool(
            q, n=4, rng=random.Random(0), route="real", passages=passages,
        )
    assert len(pool) == 4
    gens = [a.meta.get("generator") for a in pool]
    assert gens.count("real_passage") == 2
    assert gens.count("template") == 2


def test_distractor_route_real_full():
    q = Query(id="q2", domain="travel", text="问题", factual_core="问题")
    passages = [
        {"text": f"第{i}段互不相同的真实旅行内容，包含各种经验之谈。" * 10}
        for i in range(6)
    ]
    pool = make_distractor_pool(
        q, n=4, rng=random.Random(0), route="real", passages=passages,
    )
    assert len(pool) == 4
    assert all(a.meta["generator"] == "real_passage" for a in pool)
    assert len({a.id for a in pool}) == 4
