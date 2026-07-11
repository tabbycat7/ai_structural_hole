"""Hybrid (BM25 + dense bge) retriever for the Study 5 RAG loop.

A controlled target article is *injected* into a frozen real-passage corpus and
ranked against it. Two design choices keep the paired causal contrast valid and
the run cheap:

  1. Corpus term statistics (BM25 idf/avgdl) and dense embeddings are computed
     once per corpus and cached; the injected target is scored with those same
     fixed statistics. Control and treatment targets therefore face an identical
     scoring environment and differ only in their own text.
  2. Each query's corpus scores are computed once and reused across all of that
     query's targets (the 16 OFAT design points), so retrieval is fast.

Heavy deps (`jieba`, `rank_bm25`, `sentence_transformers`) are imported lazily so
importing this module never fails on a machine without them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import PATHS, RAG_EMBED_MODEL, RAG_HYBRID_ALPHA, RAG_RETRIEVER
from .corpus import CorpusDoc

# bge-zh retrieval works best with a short query instruction (s2p asymmetric).
_BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# Process-wide cache so the (heavy) embedding model is loaded exactly once, even
# when we build one retriever per domain.
_ST_MODEL_CACHE: Dict[str, object] = {}


@dataclass
class RetrievalResult:
    """Outcome of ranking one injected target against the corpus."""

    target_rank: int  # 0-based rank among corpus docs + target
    retrieved: bool  # target_rank < k
    target_score: float
    competitor_doc_ids: List[str]  # top (k-1) corpus docs (strongest real rivals)
    top_k_doc_ids: List[str]  # top-k corpus docs (for reference/logging)


@dataclass
class _QueryContext:
    query_tokens: List[str]
    bm25_corpus: Optional[np.ndarray]
    query_emb: Optional[np.ndarray]


def _tokenize(text: str) -> List[str]:
    import jieba

    return [t for t in jieba.lcut(text or "") if t.strip()]


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class HybridRetriever:
    """Rank a controlled target against one domain's frozen corpus."""

    def __init__(
        self,
        docs: Sequence[CorpusDoc],
        *,
        mode: str = RAG_RETRIEVER,
        alpha: float = RAG_HYBRID_ALPHA,
        embed_model: str = RAG_EMBED_MODEL,
        embed_cache_dir: Optional[Path] = None,
    ):
        if not docs:
            raise ValueError("HybridRetriever needs a non-empty corpus")
        self.docs = list(docs)
        self.doc_ids = [d.doc_id for d in self.docs]
        self.doc_by_id = {d.doc_id: d for d in self.docs}
        self.mode = mode
        self.alpha = alpha
        self.embed_model = embed_model
        self.embed_cache_dir = embed_cache_dir or PATHS.rag_embed_cache_dir

        self.use_bm25 = mode in ("bm25", "hybrid")
        self.use_dense = mode in ("dense", "hybrid")

        self._bm25 = None
        self._corpus_emb: Optional[np.ndarray] = None
        self._st_model = None
        self._query_cache: Dict[str, _QueryContext] = {}
        self._pre_query_emb: Dict[str, np.ndarray] = {}

        if self.use_bm25:
            self._build_bm25()
        if self.use_dense:
            self._build_dense()

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(d.text) for d in self.docs]
        self._bm25 = BM25Okapi(tokenized)

    def _corpus_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.embed_model.encode("utf-8"))
        for did in self.doc_ids:
            h.update(did.encode("utf-8"))
        return h.hexdigest()[:16]

    def _load_st_model(self):
        if self._st_model is None:
            model = _ST_MODEL_CACHE.get(self.embed_model)
            if model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.embed_model)
                _ST_MODEL_CACHE[self.embed_model] = model
            self._st_model = model
        return self._st_model

    def embed(
        self, texts: Sequence[str], *, is_query: bool = False, batch_size: int = 64
    ) -> np.ndarray:
        """Batch-encode texts into L2-normalized embeddings (public helper).

        Passing all texts at once lets sentence-transformers batch on the model,
        which is far faster than one call per document.
        """
        model = self._load_st_model()
        payload = list(texts)
        if is_query:
            payload = [_BGE_QUERY_INSTRUCTION + t for t in payload]
        emb = model.encode(
            payload,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return np.asarray(emb, dtype=np.float32)

    # Back-compat alias.
    def _encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        return self.embed(texts, is_query=is_query)

    def _build_dense(self) -> None:
        self.embed_cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self.embed_cache_dir / f"corpus_{self._corpus_fingerprint()}.npy"
        if cache.exists():
            emb = np.load(cache)
            if emb.shape[0] == len(self.docs):
                self._corpus_emb = emb
                return
        emb = self._encode([d.text for d in self.docs])
        np.save(cache, emb)
        self._corpus_emb = emb

    # ------------------------------------------------------------------ #
    # Per-query context (computed once, reused across a query's targets)
    # ------------------------------------------------------------------ #
    def warm_queries(self, query_texts: Sequence[str]) -> None:
        """Batch-encode all query embeddings up front (one model call).

        Avoids a separate encode per query during retrieval; the results feed
        `_query_ctx` via `_pre_query_emb`.
        """
        if not self.use_dense:
            return
        uniq = [t for t in dict.fromkeys(query_texts) if t not in self._pre_query_emb]
        if not uniq:
            return
        embs = self.embed(uniq, is_query=True)
        for t, e in zip(uniq, embs):
            self._pre_query_emb[t] = e

    def _query_ctx(self, query_text: str) -> _QueryContext:
        ctx = self._query_cache.get(query_text)
        if ctx is not None:
            return ctx
        tokens = _tokenize(query_text) if self.use_bm25 else []
        bm25_corpus = None
        if self.use_bm25:
            bm25_corpus = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        query_emb = None
        if self.use_dense:
            query_emb = self._pre_query_emb.get(query_text)
            if query_emb is None:
                query_emb = self.embed([query_text], is_query=True)[0]
        ctx = _QueryContext(query_tokens=tokens, bm25_corpus=bm25_corpus, query_emb=query_emb)
        self._query_cache[query_text] = ctx
        return ctx

    def _bm25_score_doc(self, query_tokens: Sequence[str], doc_text: str) -> float:
        """BM25 score of an injected doc using the *corpus* idf/avgdl stats."""
        bm25 = self._bm25
        doc_tokens = _tokenize(doc_text)
        if not doc_tokens:
            return 0.0
        freqs: Dict[str, int] = {}
        for t in doc_tokens:
            freqs[t] = freqs.get(t, 0) + 1
        dl = len(doc_tokens)
        k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl
        score = 0.0
        denom_len = k1 * (1 - b + b * dl / avgdl) if avgdl else k1
        for term in query_tokens:
            f = freqs.get(term, 0)
            if f == 0:
                continue
            idf = bm25.idf.get(term, 0.0)
            score += idf * (f * (k1 + 1)) / (f + denom_len)
        return float(score)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _fuse(self, bm25_vec: Optional[np.ndarray], dense_vec: Optional[np.ndarray]) -> np.ndarray:
        if self.use_bm25 and self.use_dense:
            return self.alpha * _minmax(dense_vec) + (1 - self.alpha) * _minmax(bm25_vec)
        if self.use_dense:
            return _minmax(dense_vec)
        return _minmax(bm25_vec)

    def _fused_scores(
        self,
        query_text: str,
        target_text: str,
        target_emb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fused score vector over [corpus_docs..., target] for one query.

        Length is len(docs) + 1; the last entry is the injected target's score.
        Corpus and target are scored with the *same* frozen statistics, so a
        control/treatment pair differs only in the target's own text.
        """
        ctx = self._query_ctx(query_text)

        bm25_all = dense_all = None
        if self.use_bm25:
            target_bm25 = self._bm25_score_doc(ctx.query_tokens, target_text)
            bm25_all = np.concatenate([ctx.bm25_corpus, [target_bm25]])
        if self.use_dense:
            if target_emb is None:
                target_emb = self.embed([target_text])[0]
            corpus_sims = self._corpus_emb @ ctx.query_emb
            target_sim = float(target_emb @ ctx.query_emb)
            dense_all = np.concatenate([corpus_sims, [target_sim]])

        return self._fuse(bm25_all, dense_all)

    def _result_from_fused(self, fused: np.ndarray, k: int) -> RetrievalResult:
        n = len(self.docs)
        corpus_scores = fused[:n]
        target_score = float(fused[n])

        # Rank: number of corpus docs scoring strictly higher than the target.
        target_rank = int(np.sum(corpus_scores > target_score))

        order = np.argsort(-corpus_scores, kind="stable")
        ranked_ids = [self.doc_ids[i] for i in order]
        top_k_ids = ranked_ids[:k]
        competitor_ids = ranked_ids[: max(k - 1, 0)]

        return RetrievalResult(
            target_rank=target_rank,
            retrieved=target_rank < k,
            target_score=target_score,
            competitor_doc_ids=competitor_ids,
            top_k_doc_ids=top_k_ids,
        )

    def retrieve(
        self,
        query_text: str,
        target_text: str,
        k: int,
        *,
        target_emb: Optional[np.ndarray] = None,
    ) -> RetrievalResult:
        """Rank the injected target against the corpus.

        `target_emb` (optional) lets the caller pass a precomputed dense
        embedding (e.g. from a batched `embed(...)` over all targets), avoiding a
        per-target model call.
        """
        fused = self._fused_scores(query_text, target_text, target_emb)
        return self._result_from_fused(fused, k)

    def retrieve_multi(
        self,
        query_texts: Sequence[str],
        target_text: str,
        k: int,
        *,
        target_emb: Optional[np.ndarray] = None,
        fuse: str = "max",
    ) -> RetrievalResult:
        """Rank the target using several rewritten queries, pooled per document.

        Each query yields a fused score vector over [corpus..., target]; those
        vectors are combined element-wise by `max` (default) or `mean`. A single
        query reduces exactly to `retrieve`. Empty input falls back to the target
        text itself as the query (keeps the run defined).
        """
        queries = [q for q in (query_texts or []) if str(q).strip()]
        if not queries:
            queries = [target_text]
        if fuse not in ("max", "mean"):
            raise ValueError(f"unknown fuse mode: {fuse}")

        vecs = [
            self._fused_scores(q, target_text, target_emb) for q in queries
        ]
        stacked = np.vstack(vecs)
        pooled = stacked.max(axis=0) if fuse == "max" else stacked.mean(axis=0)
        return self._result_from_fused(pooled, k)
