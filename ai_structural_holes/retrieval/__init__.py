"""Real RAG retrieval loop (Study 5).

This subpackage upgrades the "forced single-choice among 3 inlined candidates"
task into a real two-stage RAG pipeline:

  - `corpus`   : freeze a per-domain corpus from the real query-pool passages.
  - `retriever`: a hybrid BM25 + dense (bge) retriever that ranks a controlled
    target article against that real corpus (the retrieval channel of the
    structural hole).

The generation channel (does the model cite the target once it is in context)
reuses the existing task protocol / experiment runner with a citation output
mode.
"""
from __future__ import annotations

from .corpus import CorpusDoc, build_corpus, corpus_path, load_corpus
from .retriever import HybridRetriever, RetrievalResult

__all__ = [
    "CorpusDoc",
    "build_corpus",
    "corpus_path",
    "load_corpus",
    "HybridRetriever",
    "RetrievalResult",
]
