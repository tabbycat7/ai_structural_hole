"""Group Study 4 audit rows by query for manual review."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .query_pool import load_pool_records

_DIM_ORDER = {"S1": 0, "S3": 1}
_VARIANT_ORDER = {"none": 0, "genuine": 1, "fake": 2}

_DOMAIN_LABEL = {
    "consumer_product": "消费",
    "health": "健康",
    "finance": "金融",
    "academic_qa": "学术",
    "travel": "旅行",
}


def _query_text_map() -> dict[str, str]:
    records = load_pool_records()
    out: dict[str, str] = {}
    for qid, rec in records.items():
        out[qid] = (rec.get("question") or rec.get("factual_core") or "").strip()
    return out


def group_audit_sheet_for_review(
    audit_df: pd.DataFrame,
    *,
    verification_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Sort audit rows so all variants of one query appear together.

    Order: domain → query_id → dim (S1, S3) → variant (none, genuine, fake).
    Adds `group_no`, `query_text`, and optional verification columns.
    """
    df = audit_df.copy()
    qtext = _query_text_map()
    df["query_text"] = df["query_id"].map(qtext).fillna("")

    if verification_df is not None and not verification_df.empty:
        ver_cols = [
            "article_id",
            "verification_status",
            "primary_title",
            "evidence",
            "recommended_verdict",
        ]
        ver = verification_df[[c for c in ver_cols if c in verification_df.columns]]
        df = df.merge(ver, on="article_id", how="left", suffixes=("", "_ver"))

    df["_dim_ord"] = df["dim"].map(_DIM_ORDER).fillna(9)
    df["_var_ord"] = df["variant"].map(_VARIANT_ORDER).fillna(9)
    df = df.sort_values(
        ["domain", "query_id", "_dim_ord", "_var_ord", "article_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    # stable group number within sorted order
    df["group_no"] = pd.factorize(df["query_id"], sort=False)[0] + 1
    df["domain_label"] = df["domain"].map(_DOMAIN_LABEL).fillna(df["domain"])
    df["group_label"] = (
        "G" + df["group_no"].astype(str).str.zfill(3)
        + " | " + df["domain_label"]
        + " | " + df["query_id"].str[:12]
        + " | " + df["query_text"].str.slice(0, 60)
    )

    front = [
        "group_no",
        "group_label",
        "query_id",
        "domain",
        "domain_label",
        "query_text",
        "dim",
        "variant",
        "generator",
        "n_chars",
    ]
    ver_front = [
        "verification_status",
        "primary_title",
        "evidence",
        "recommended_verdict",
    ]
    back = [
        "text",
        "verifiable",
        "source_url_or_doi",
        "verdict",
        "reviewer",
        "note",
        "article_id",
    ]
    cols = [c for c in front + ver_front + back if c in df.columns]
    out = df[cols].copy()
    return out


def write_audit_sheet_review(
    audit_path: Path,
    out_path: Path,
    *,
    verification_path: Optional[Path] = None,
) -> Path:
    audit = pd.read_csv(audit_path)
    ver = None
    if verification_path is not None and verification_path.exists():
        ver = pd.read_csv(verification_path)
    grouped = group_audit_sheet_for_review(audit, verification_df=ver)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
