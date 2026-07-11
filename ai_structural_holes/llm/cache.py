"""On-disk response cache keyed by a hash of the request.

Avoids paying twice for identical (model, messages, params) requests and makes
runs reproducible. Each entry is a small JSON file under the cache dir.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union


def request_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class NullDiskCache:
    """No-op cache: always miss, never writes."""

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        pass


class DiskCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        # Atomic write (temp file + os.replace) so concurrent writers can never
        # be observed reading a half-written entry.
        p = self._path(key)
        tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)


CacheBackend = Union[DiskCache, NullDiskCache]


def resolve_llm_cache(use_llm_cache: Optional[bool] = None) -> CacheBackend:
    """Return DiskCache when enabled, else NullDiskCache (study default)."""
    from ..config import PATHS, llm_cache_enabled

    enabled = llm_cache_enabled() if use_llm_cache is None else bool(use_llm_cache)
    return DiskCache(PATHS.cache_dir) if enabled else NullDiskCache()
