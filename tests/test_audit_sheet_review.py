import pandas as pd

from ai_structural_holes.data.audit_sheet_review import group_audit_sheet_for_review


def test_group_audit_sheet_orders_variants():
    audit = pd.DataFrame([
        {"article_id": "a3", "query_id": "q1", "domain": "health", "dim": "S3",
         "variant": "fake", "generator": "llm", "n_chars": 100, "text": "t3"},
        {"article_id": "a1", "query_id": "q1", "domain": "health", "dim": "S1",
         "variant": "genuine", "generator": "llm", "n_chars": 100, "text": "t1"},
        {"article_id": "a2", "query_id": "q1", "domain": "health", "dim": "S1",
         "variant": "none", "generator": "llm", "n_chars": 100, "text": "t2"},
    ])
    out = group_audit_sheet_for_review(audit)
    sub = out[out["query_id"] == "q1"]
    assert list(sub["variant"]) == ["none", "genuine", "fake"]
