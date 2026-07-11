"""Tests for variant store abnormal scan / classify."""
from ai_structural_holes.data.variant_repair import classify_record, scan_abnormal_records


def test_classify_short_and_truncated():
    base = "x" * 400
    bases = {"q1": base}
    ok = {"article_id": "a", "query_id": "q1", "generator": "llm", "text": "y" * 200}
    assert classify_record(ok, bases, min_chars=50) is None

    short = {"article_id": "b", "query_id": "q1", "generator": "llm", "text": "短"}
    assert classify_record(short, bases, min_chars=50) == "short(1<50)"

    trunc = {"article_id": "c", "query_id": "q1", "generator": "llm", "text": "y" * 40}
    assert classify_record(trunc, bases, min_chars=10, min_base_ratio=0.15) == "truncated(40/400)"


def test_scan_abnormal_records():
    store = {
        "good": {"article_id": "good", "query_id": "q", "generator": "llm", "text": "a" * 100, "is_target": True},
        "bad": {"article_id": "bad", "query_id": "q", "generator": "llm", "text": "短", "is_target": False},
    }
    hits = scan_abnormal_records(store, {"q": "b" * 300}, min_chars=50)
    assert len(hits) == 1
    assert hits[0].article_id == "bad"
