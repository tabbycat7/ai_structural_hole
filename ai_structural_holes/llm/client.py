"""Unified model-call layer routed through OpenRouter.

All models share one OpenAI-compatible endpoint; switching model = switching the
`model` slug. Features:
  - disk cache keyed by request hash (dedup, reproducibility, cost control)
  - retry with exponential backoff on transient/rate-limit errors (tenacity)
  - usage logging (tokens) returned with every response
  - a `MockClient` so the whole pipeline runs offline / in tests without a key

Use `get_client(mock=...)` to obtain a client. With no API key present it falls
back to the mock automatically (with a warning) so nothing crashes offline.
"""
from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import PATHS
from .cache import DiskCache, request_hash

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:  # pragma: no cover - tenacity optional at import time
    retry = None


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[dict] = None


class BaseClient:
    def call(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = 800,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenRouterClient(BaseClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[DiskCache] = None,
        max_attempts: int = 5,
        request_timeout: float = 60.0,
    ):
        from openai import OpenAI  # imported lazily so offline import works

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        default_headers = {}
        if os.environ.get("OPENROUTER_HTTP_REFERER"):
            default_headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
        if os.environ.get("OPENROUTER_X_TITLE"):
            default_headers["X-Title"] = os.environ["OPENROUTER_X_TITLE"]
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
            timeout=request_timeout,
            default_headers=default_headers or None,
        )
        self.cache = cache or DiskCache(PATHS.cache_dir)
        self.max_attempts = max_attempts

    def _raw_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._client.chat.completions.create(**payload)
        text = resp.choices[0].message.content or ""
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return {"text": text, "usage": usage}

    def call(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = 800,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        key = request_hash(payload)
        cached = self.cache.get(key)
        if cached is not None:
            return LLMResponse(
                text=cached["text"], model=model, cached=True, usage=cached.get("usage", {})
            )

        result = self._call_with_retry(payload)
        self.cache.set(key, result)
        return LLMResponse(text=result["text"], model=model, cached=False, usage=result.get("usage", {}))

    def _call_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if retry is None:
            return self._raw_call(payload)

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def _do():
            return self._raw_call(payload)

        return _do()


# Transparent content-marker weights used by the MockClient to fabricate a
# plausible (but synthetic) selection behaviour driven by both position and
# content. NOT a model of any real LLM.
_MOCK_CONTENT_WEIGHTS = {
    "%": 6.0,           # S1 evidence (statistics)
    "机构": 5.0,         # S1 source
    "核心优势": 7.0,      # S4 explicit claim
    "但是": 3.0,         # S2 balance
    "局限": 3.0,
    "机制": 4.0,         # S3 expertise
    "(证据:": 5.0,       # O4 binding
    "\n- ": 3.0,        # O1 list
    "##": 2.5,          # O3 headings
    "结论先行": 2.5,      # O2 conclusion-first
}


class MockClient(BaseClient):
    """Deterministic offline client.

    Produces a syntactically valid JSON decision. Selection is a transparent
    function of (a) candidate *position* (earlier favoured -> position bias) and
    (b) a content score from feature markers, plus request-seeded noise. This
    exercises the full pipeline (parsing, ATE, backdoor, EI) with a known signal
    so content dimensions get non-zero EI. It is NOT a model of real LLM behaviour.
    """

    def __init__(self, position_bias: float = 0.85, content_gain: float = 1.0,
                 noise: float = 8.0):
        self.position_bias = position_bias
        self.content_gain = content_gain
        self.noise = noise

    @staticmethod
    def _split_candidates(user: str):
        import re

        parts = re.split(r"【候选 ([A-Z])】", user)
        # parts = [pre, 'A', textA, 'B', textB, ...]
        out = []
        for i in range(1, len(parts) - 1, 2):
            out.append((parts[i], parts[i + 1]))
        return out

    def _content_score(self, text: str) -> float:
        return self.content_gain * sum(
            w * text.count(m) for m, w in _MOCK_CONTENT_WEIGHTS.items()
        )

    def call(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = 800,
    ) -> LLMResponse:
        import json as _json

        user = messages[-1]["content"]
        cands = self._split_candidates(user)
        if not cands:
            cands = [("A", user)]
        rng = random.Random(request_hash({"m": model, "u": user, "s": seed})[:8])
        scores = {}
        for i, (ltr, text) in enumerate(cands):
            position_component = 40.0 * (self.position_bias ** i)
            content_component = self._content_score(text)
            scores[ltr] = max(1.0, position_component + content_component + rng.uniform(-self.noise, self.noise))
        letters = [c[0] for c in cands]
        ranking = sorted(letters, key=lambda l: -scores[l])
        choice = ranking[0]
        payload = {
            "choice": choice,
            "ranking": ranking,
            "scores": {l: round(scores[l], 1) for l in letters},
            "reason": "mock decision (position + content)",
        }
        return LLMResponse(text=_json.dumps(payload, ensure_ascii=False), model=model, cached=False)


def get_client(mock: Optional[bool] = None) -> BaseClient:
    """Return an OpenRouter client, or a mock when no key / mock requested."""
    if mock is True:
        return MockClient()
    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    if mock is None and not has_key:
        warnings.warn("OPENROUTER_API_KEY not set; falling back to MockClient.")
        return MockClient()
    if mock is False and not has_key:
        raise RuntimeError("mock=False but OPENROUTER_API_KEY not set")
    return OpenRouterClient()
