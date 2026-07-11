"""Scan and repair abnormal entries in the frozen variant article store.

Typical failures after bulk LLM generation:
  - truncated outputs (a few characters / one sentence fragment)
  - empty or near-empty text that was cached before the client stopped
    persisting blank responses
  - non-llm generator flags (template_fallback) still present in the store

`repair_abnormal_variants` rewrites only the flagged records via
`llm_edit_variant` (with a per-article seed offset to bypass stale truncated
cache entries) and writes the results back into `data/variant_articles/`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .base_articles import load_base_articles, load_base_texts
from .generation import llm_edit_variant
from .query_pool import load_pool_records
from .schema import Article, normalize_profile
from .variant_articles import load_variant_store, record_from_article, save_variant_records


@dataclass
class AbnormalRecord:
    article_id: str
    query_id: str
    reason: str
    n_chars: int
    is_target: bool
    gen_model: str
    text_preview: str


def _seed_offset_for_regen(article_id: str) -> int:
    """Stable cache-bust offset so regen does not replay a truncated cache hit."""
    h = int(hashlib.sha256(article_id.encode("utf-8")).hexdigest()[:8], 16)
    return 10_000 + (h % 90_000)


def classify_record(
    rec: dict,
    base_texts: Dict[str, str],
    *,
    min_chars: int = 50,
    min_base_ratio: float = 0.15,
) -> Optional[str]:
    """Return a human-readable reason if `rec` is abnormal, else None."""
    text = (rec.get("text") or "").strip()
    n = len(text)
    if rec.get("generator") != "llm":
        return f"generator={rec.get('generator')}"
    if n == 0:
        return "empty"
    if n < min_chars:
        return f"short({n}<{min_chars})"
    base = base_texts.get(rec.get("query_id") or "", "")
    if base and n < int(len(base) * min_base_ratio):
        return f"truncated({n}/{len(base)})"
    return None


def scan_abnormal_records(
    store: Optional[Dict[str, dict]] = None,
    base_texts: Optional[Dict[str, str]] = None,
    *,
    min_chars: int = 50,
    min_base_ratio: float = 0.15,
) -> List[AbnormalRecord]:
    """Scan the full variant store and return every abnormal record."""
    store = store if store is not None else load_variant_store()
    base_texts = base_texts if base_texts is not None else load_base_texts()
    out: List[AbnormalRecord] = []
    for rec in store.values():
        reason = classify_record(
            rec, base_texts, min_chars=min_chars, min_base_ratio=min_base_ratio,
        )
        if reason is None:
            continue
        text = (rec.get("text") or "").strip()
        out.append(
            AbnormalRecord(
                article_id=rec["article_id"],
                query_id=rec["query_id"],
                reason=reason,
                n_chars=len(text),
                is_target=bool(rec.get("is_target")),
                gen_model=str(rec.get("gen_model") or ""),
                text_preview=text[:60].replace("\n", " "),
            )
        )
    out.sort(key=lambda r: (r.n_chars, r.article_id))
    return out


def _query_core(query_id: str, pool: Dict[str, dict], base_meta: Dict[str, dict]) -> str:
    if query_id in pool:
        rec = pool[query_id]
        return rec.get("factual_core") or rec.get("question") or ""
    meta = base_meta.get(query_id) or {}
    return meta.get("factual_core") or meta.get("query_text") or ""


def _repair_one(
    rec: dict,
    *,
    client,
    base_texts: Dict[str, str],
    pool: Dict[str, dict],
    base_meta: Dict[str, dict],
    gen_model: str,
    min_chars: int,
    min_base_ratio: float,
    max_attempts: int,
    regen_pass: int = 0,
) -> Tuple[Optional[dict], str]:
    """Regenerate one store record; return (new_record, status)."""
    aid = rec["article_id"]
    qid = rec["query_id"]
    base_text = base_texts.get(qid)
    if not base_text:
        return None, "no_base"
    model = gen_model or rec.get("gen_model") or ""
    if not model:
        return None, "no_model"

    min_len = max(min_chars, int(len(base_text) * min_base_ratio))

    edited, report = llm_edit_variant(
        client,
        base_text=base_text,
        target_profile=normalize_profile(rec.get("intended_profile") or {}),
        core=_query_core(qid, pool, base_meta),
        model=model,
        authenticity=rec.get("authenticity") or "genuine",
        max_attempts=max_attempts,
        min_chars=min_len,
        seed_offset=_seed_offset_for_regen(aid) + regen_pass * 1_000,
    )
    if edited is None:
        return None, f"failed({report.get('n_chars', 0)}ch)"
    if classify_record(
        {"text": edited, "generator": "llm", "query_id": qid},
        base_texts,
        min_chars=min_chars,
        min_base_ratio=min_base_ratio,
    ):
        return None, f"still_bad({len(edited)}ch)"

    art = Article(
        id=aid,
        query_id=qid,
        text=edited,
        is_target=bool(rec.get("is_target")),
        authenticity=rec.get("authenticity") or "genuine",
        intended_profile=normalize_profile(rec.get("intended_profile") or {}),
        meta={"generator": "llm", "edit_attempts": report.get("attempts")},
    )
    return record_from_article(art, base_text, model), "ok"


def repair_abnormal_variants(
    *,
    client,
    gen_model: Optional[str] = None,
    min_chars: int = 50,
    min_base_ratio: float = 0.15,
    max_attempts: int = 5,
    max_passes: int = 2,
    concurrency: int = 4,
    progress: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Find abnormal variants in the store, regenerate them, and persist fixes.

    Returns a summary dict with counts and per-article status rows.
    """
    from ..llm.parallel import map_concurrent
    from ..studies.common import make_progress_bar

    store = load_variant_store()
    base_texts = load_base_texts()
    base_meta = load_base_articles(validated_only=False)
    pool = load_pool_records()

    abnormal = scan_abnormal_records(
        store, base_texts, min_chars=min_chars, min_base_ratio=min_base_ratio,
    )
    summary: Dict[str, object] = {
        "scanned": len(store),
        "abnormal": len(abnormal),
        "min_chars": min_chars,
        "min_base_ratio": min_base_ratio,
        "dry_run": dry_run,
        "rows": [],
    }
    if not abnormal:
        return summary

    if dry_run:
        summary["rows"] = [a.__dict__ for a in abnormal]
        return summary

    to_fix = [store[a.article_id] for a in abnormal]
    bar = make_progress_bar(len(to_fix), desc="修复异常变体", enabled=progress)
    status_rows: List[dict] = []
    all_new_records: List[dict] = []

    def _run_pass(records: List[dict], regen_pass: int) -> List[dict]:
        """Regenerate `records`; return those still failing after this pass."""
        still: List[dict] = []

        def _job(rec):
            new_rec, status = _repair_one(
                rec,
                client=client,
                base_texts=base_texts,
                pool=pool,
                base_meta=base_meta,
                gen_model=gen_model or rec.get("gen_model", ""),
                min_chars=min_chars,
                min_base_ratio=min_base_ratio,
                max_attempts=max_attempts,
                regen_pass=regen_pass,
            )
            return rec, new_rec, status

        def _on(_i, rec, res):
            old, new_rec, status = res
            old_n = len((old.get("text") or "").strip())
            new_n = len((new_rec or {}).get("text") or "") if new_rec else 0
            status_rows.append({
                "article_id": old["article_id"],
                "query_id": old["query_id"],
                "is_target": old.get("is_target"),
                "status": status,
                "pass": regen_pass,
                "old_n_chars": old_n,
                "new_n_chars": new_n,
                "old_preview": (old.get("text") or "")[:40],
                "new_preview": ((new_rec or {}).get("text") or "")[:40],
            })
            if bar is not None:
                bar.set_postfix_str(f"p{regen_pass} {status} {old_n}->{new_n}")
                bar.update(1)
            if new_rec is None or status != "ok":
                still.append(old)
            else:
                all_new_records.append(new_rec)

        map_concurrent(_job, records, concurrency=concurrency, on_result=_on)
        return still

    pending = to_fix
    for regen_pass in range(max_passes):
        if not pending:
            break
        if regen_pass > 0 and bar is not None:
            bar.total += len(pending)
        pending = _run_pass(pending, regen_pass)

    if bar is not None:
        bar.close()

    ok = len(all_new_records)
    failed = len(pending)

    if all_new_records:
        save_variant_records(all_new_records)

    summary["fixed"] = ok
    summary["still_bad"] = failed
    summary["rows"] = status_rows
    return summary
