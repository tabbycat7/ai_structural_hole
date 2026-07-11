"""Frozen per-domain retrieval corpus built from real query-pool passages.

Study 5 needs a realistic competition environment: instead of 2-3 hand-picked
distractors, a controlled target article competes against a large index of real
web passages. We reuse the passages already frozen in `data/query_pool/` (see
`data.query_pool`) and aggregate them, per domain, into a deduplicated corpus
that is frozen to `data/rag_corpus/<domain>.jsonl` (commit it!).

Freezing discipline mirrors `data/base_articles/` and `data/query_pool/`: build
once, spot-check by hand, never regenerate mid-experiment. Retrieval over a
frozen corpus is deterministic, so the control/treatment contrast differs only
in the injected target text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..config import DOMAINS, PATHS
from ..data.schema import stable_id

_WS_RE = re.compile(r"\s+")


@dataclass
class CorpusDoc:
    """One retrievable document in the frozen corpus."""

    doc_id: str
    domain: str
    text: str
    src_query_id: str
    n_chars: int = 0

    def __post_init__(self):
        if not self.n_chars:
            self.n_chars = len(self.text)


def corpus_dir(root: Optional[Path] = None) -> Path:
    d = (root or PATHS.data_dir) / "rag_corpus"
    d.mkdir(parents=True, exist_ok=True)
    return d


def corpus_path(domain: str, root: Optional[Path] = None) -> Path:
    return corpus_dir(root) / f"{domain}.jsonl"


def _dedup_key(text: str) -> str:
    """Stable near-dup key: first 60 non-space chars (same policy as query_pool)."""
    return _WS_RE.sub("", text)[:60]


def build_corpus(
    domains: Optional[Sequence[str]] = None,
    root: Optional[Path] = None,
    pool_root: Optional[Path] = None,
) -> Dict[str, int]:
    """Aggregate + dedup real pool passages per domain and freeze to jsonl.

    Returns a per-domain document count. Deterministic given the frozen pool
    (documents are emitted in sorted query-id order, dedup keeps the first).
    """
    from ..data.query_pool import load_pool_records

    domains = list(domains or DOMAINS)
    records = load_pool_records(pool_root)

    by_domain: Dict[str, List[CorpusDoc]] = {d: [] for d in domains}
    seen: Dict[str, set] = {d: set() for d in domains}

    for qid in sorted(records.keys()):
        rec = records[qid]
        dom = rec.get("domain")
        if dom not in by_domain:
            continue
        for passage in rec.get("passages", []):
            text = (passage.get("text") or "").strip()
            if not text:
                continue
            key = _dedup_key(text)
            if key in seen[dom]:
                continue
            seen[dom].add(key)
            doc_id = stable_id("corpusdoc", dom, key)
            by_domain[dom].append(
                CorpusDoc(doc_id=doc_id, domain=dom, text=text, src_query_id=qid)
            )

    counts: Dict[str, int] = {}
    for dom in domains:
        docs = by_domain[dom]
        path = corpus_path(dom, root)
        body = "\n".join(json.dumps(asdict(d), ensure_ascii=False) for d in docs)
        path.write_text(body + ("\n" if body else ""), encoding="utf-8")
        counts[dom] = len(docs)
    return counts


def load_corpus(domain: str, root: Optional[Path] = None) -> List[CorpusDoc]:
    """Load one domain's frozen corpus (empty list if not built yet)."""
    path = corpus_path(domain, root)
    if not path.exists():
        return []
    docs: List[CorpusDoc] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        docs.append(
            CorpusDoc(
                doc_id=obj["doc_id"],
                domain=obj.get("domain", domain),
                text=obj["text"],
                src_query_id=obj.get("src_query_id", ""),
                n_chars=obj.get("n_chars", 0),
            )
        )
    return docs
