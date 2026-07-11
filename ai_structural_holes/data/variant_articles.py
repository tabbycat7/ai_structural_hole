"""Frozen LLM-edited *variant* articles (the reusable target/distractor cache).

A *variant article* is the LLM rewrite of a query's frozen base article into a
specific S/O target profile (see `generation.llm_edit_variant`). Generating the
full experiment can mean thousands of such rewrites, so we persist each
successful one to `data/variant_articles/<query_id>.jsonl` (one line per
variant) and reuse it on later runs. Unlike the low-level request cache in
`.cache/llm/`, this store is human-auditable (readable prose per query) and
git-committable (freeze the exact experimental material).

Reuse is guarded by a fingerprint: a stored variant is only reused when its
`article_id`, `gen_model`, and `base_hash` all match the current run, so a
changed base article or a different rewrite model transparently invalidates the
old record and triggers regeneration. Only successful `generator == "llm"`
variants are stored; `template_fallback` results are left out so they get
another chance to succeed next time.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from ..config import PATHS
from .schema import Article


def variant_dir(root: Optional[Path] = None) -> Path:
    d = (root or PATHS.data_dir) / "variant_articles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def variant_path(query_id: str, root: Optional[Path] = None) -> Path:
    return variant_dir(root) / f"{query_id}.jsonl"


def base_hash(base_text: str) -> str:
    """Stable fingerprint of the base article a variant was rewritten from."""
    return hashlib.sha256((base_text or "").encode("utf-8")).hexdigest()


def load_variant_store(root: Optional[Path] = None) -> Dict[str, dict]:
    """Load all frozen variant records keyed by `article_id`.

    Scans every `<query_id>.jsonl` under the store; on duplicate ids the last
    line wins (matches the append/rewrite semantics of `save_variant_records`).
    """
    out: Dict[str, dict] = {}
    d = variant_dir(root)
    for path in sorted(d.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            aid = rec.get("article_id")
            if aid:
                out[aid] = rec
    return out


def record_from_article(article: Article, base_text: str, gen_model: str) -> dict:
    """Serialize a finalized LLM variant article into a store record."""
    meta = article.meta or {}
    verified = article.verified_profile
    return {
        "article_id": article.id,
        "query_id": article.query_id,
        "gen_model": gen_model,
        "base_hash": base_hash(base_text),
        "authenticity": article.authenticity,
        "is_target": article.is_target,
        "intended_profile": dict(article.intended_profile),
        "verified_profile": dict(verified) if verified else None,
        "generator": meta.get("generator"),
        "edit_attempts": meta.get("edit_attempts"),
        "text": article.text,
        "n_chars": article.n_chars,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def is_hit(record: dict, *, gen_model: str, base_text: str) -> bool:
    """Whether `record` is a valid reuse for the current base text / model."""
    return (
        record.get("generator") == "llm"
        and record.get("gen_model") == gen_model
        and record.get("base_hash") == base_hash(base_text)
    )


def apply_record(article: Article, record: dict) -> Article:
    """Fill `article` in place from a stored variant record (a store 'hit').

    Mirrors `generation.finalize_llm_article` for the reuse path so downstream
    behaviour is identical to a freshly generated llm variant.
    """
    article.text = record["text"]
    article.n_chars = record.get("n_chars") or len(article.text)
    verified = record.get("verified_profile")
    article.verified_profile = dict(verified) if verified else None
    article.manipulation_ok = True
    meta = article.meta
    meta["generator"] = record.get("generator", "llm")
    meta["edit_attempts"] = record.get("edit_attempts")
    return article


def save_variant_records(
    records: List[dict], root: Optional[Path] = None
) -> List[Path]:
    """Persist variant records, grouped per query, merging with existing files.

    Same `article_id` overwrites the prior record; each query's jsonl is written
    atomically (temp file + os.replace) so a crashed/concurrent run can never be
    observed reading a half-written file.
    """
    if not records:
        return []

    by_query: Dict[str, List[dict]] = {}
    for rec in records:
        qid = rec.get("query_id")
        if not qid:
            continue
        by_query.setdefault(qid, []).append(rec)

    written: List[Path] = []
    for qid, new_recs in by_query.items():
        path = variant_path(qid, root)
        merged: Dict[str, dict] = {}
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("article_id"):
                        merged[rec["article_id"]] = rec
            except Exception:
                merged = {}
        for rec in new_recs:
            merged[rec["article_id"]] = rec

        body = "\n".join(
            json.dumps(rec, ensure_ascii=False) for rec in merged.values()
        )
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(body + "\n", encoding="utf-8")
        os.replace(tmp, path)
        written.append(path)
    return written
