"""Unified model-call layer with provider routing.

Routing:
  - deepseek/*  -> DeepSeek official API (https://api.deepseek.com)
  - qwen/*, qwen* -> Alibaba Cloud MaaS compatible endpoint (QWEN_BASE_URL)
  - doubao/*, doubao* -> Volcengine Ark API (DOUBAO_BASE_URL)
  - kimi/*, kimi*, moonshot/*, moonshot* -> Moonshot API (KIMI_BASE_URL)
  - minimax/*, minimax*, abab* -> MiniMax API (MINIMAX_BASE_URL)
  - everything else -> OpenRouter

Both provider endpoints are OpenAI-compatible.

Features:
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
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import PATHS
from .cache import CacheBackend, DiskCache, NullDiskCache, resolve_llm_cache, request_hash

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:  # pragma: no cover - tenacity optional at import time
    retry = None


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "https://llm-mvkibj7hczl2nxnk.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
DOUBAO_BASE_URL = os.environ.get(
    "DOUBAO_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3",
)
KIMI_BASE_URL = os.environ.get(
    "KIMI_BASE_URL",
    "https://api.moonshot.cn/v1",
)
MINIMAX_BASE_URL = os.environ.get(
    "MINIMAX_BASE_URL",
    "https://api.minimaxi.com/v1",
)


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


def is_deepseek_model(model: str) -> bool:
    return model.startswith("deepseek/") or model.startswith("deepseek-")


def resolve_deepseek_model(model: str) -> str:
    """Map roster slugs (e.g. deepseek/deepseek-chat) to DeepSeek API model ids."""
    if model.startswith("deepseek/"):
        return model.split("/", 1)[1]
    return model


def is_qwen_model(model: str) -> bool:
    """True for qwen/* slugs and bare ids like qwen3.6-flash."""
    m = model.lower()
    return m.startswith("qwen/") or m.startswith("qwen")


def resolve_qwen_model(model: str) -> str:
    """Map roster slugs (e.g. qwen/qwen3.6-flash) to the MaaS API model id."""
    if model.lower().startswith("qwen/"):
        return model.split("/", 1)[1]
    return model


def is_doubao_model(model: str) -> bool:
    """True for doubao/* slugs and bare ids like doubao-seed-2-0-mini-260428."""
    m = model.lower()
    return m.startswith("doubao/") or m.startswith("doubao")


def resolve_doubao_model(model: str) -> str:
    """Map roster slugs (e.g. doubao/doubao-seed-2-0-mini-260428) to the Ark API model id."""
    if model.lower().startswith("doubao/"):
        return model.split("/", 1)[1]
    return model


def is_kimi_model(model: str) -> bool:
    """True for kimi/*, kimi*, moonshot/* and moonshot-* slugs."""
    m = model.lower()
    if m.startswith("kimi/") or m.startswith("kimi"):
        return True
    return m.startswith("moonshot/") or m.startswith("moonshot-")


def resolve_kimi_model(model: str) -> str:
    """Map roster slugs (e.g. kimi/kimi-k2) to the Moonshot API model id."""
    m = model.lower()
    if m.startswith("kimi/") or m.startswith("moonshot/"):
        return model.split("/", 1)[1]
    return model


def is_minimax_model(model: str) -> bool:
    """True for minimax/*, minimax-* bare ids, and abab* slugs."""
    m = model.lower()
    if m.startswith("minimax/") or m.startswith("abab"):
        return True
    if not m.startswith("minimax"):
        return False
    rest = m[len("minimax") :]
    return not rest or rest[0] in "-_."


def resolve_minimax_model(model: str) -> str:
    """Map roster slugs (e.g. minimax/abab6.5s-chat) to the MiniMax API model id."""
    if model.lower().startswith("minimax/"):
        return model.split("/", 1)[1]
    return model


def minimax_thinking_can_disable(model_id: str) -> bool:
    """M3 models support thinking.type=disabled; M2.x cannot turn thinking off."""
    m = model_id.lower()
    return "m3" in m or m.startswith("minimax-m3")


def minimax_disable_thinking() -> bool:
    """Env MINIMAX_DISABLE_THINKING defaults to on (1/true/yes)."""
    v = os.environ.get("MINIMAX_DISABLE_THINKING", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _strip_thinking_markup(text: str) -> str:
    """Remove inline thinking blocks (MiniMax/Kimi) before JSON parsing."""
    import re

    return re.sub(
        r"<think>.*?</think>\s*",
        "",
        text or "",
        flags=re.DOTALL,
    ).strip()


def kimi_thinking_can_disable(model_id: str) -> bool:
    """True for thinking-capable Kimi ids except k2.7-code (always on, rejects disabled)."""
    m = model_id.lower()
    if "k2.7-code" in m:
        return False
    return "k2.6" in m or "k2.5" in m or m.startswith("kimi-k2") or m == "kimi-k2"


def kimi_disable_thinking() -> bool:
    """Env KIMI_DISABLE_THINKING defaults to on (1/true/yes)."""
    v = os.environ.get("KIMI_DISABLE_THINKING", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def kimi_omit_temperature(model_id: str) -> bool:
    """k2.5/k2.6/k2.7-code reject explicit temperature (only default 1 allowed)."""
    m = model_id.lower()
    if "k2.7-code" in m:
        return True
    return "k2.5" in m or "k2.6" in m or m.startswith("kimi-k2") or m == "kimi-k2"


class OpenAICompatibleClient(BaseClient):
    """OpenAI SDK client against any compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        cache: Optional[CacheBackend] = None,
        max_attempts: int = 5,
        request_timeout: float = 60.0,
        default_headers: Optional[Dict[str, str]] = None,
        provider: str = "openai-compatible",
    ):
        from openai import OpenAI  # imported lazily so offline import works

        if not api_key:
            raise RuntimeError(f"{provider} API key not set")
        self.provider = provider
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            default_headers=default_headers or None,
        )
        self.cache = cache if cache is not None else resolve_llm_cache()
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
        payload = self._augment_payload(payload)

        key = request_hash({**payload, "provider": self.provider})
        cached = self.cache.get(key)
        if cached is not None:
            return LLMResponse(
                text=cached["text"], model=model, cached=True, usage=cached.get("usage", {})
            )

        result = self._call_with_retry(payload)
        # Never persist empty responses: a cached blank would lock in API failures
        # across regen runs (see regen-variants / variant_repair).
        if (result.get("text") or "").strip():
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

    def _augment_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload


class OpenRouterClient(OpenAICompatibleClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[CacheBackend] = None,
        max_attempts: int = 5,
        request_timeout: float = 60.0,
    ):
        default_headers = {}
        if os.environ.get("OPENROUTER_HTTP_REFERER"):
            default_headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
        if os.environ.get("OPENROUTER_X_TITLE"):
            default_headers["X-Title"] = os.environ["OPENROUTER_X_TITLE"]
        super().__init__(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            cache=cache,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            default_headers=default_headers or None,
            provider="openrouter",
        )


class DeepSeekClient(OpenAICompatibleClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[CacheBackend] = None,
        max_attempts: int = 5,
        request_timeout: float = 60.0,
    ):
        super().__init__(
            base_url=DEEPSEEK_BASE_URL,
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            cache=cache,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            provider="deepseek",
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
        return super().call(
            model=resolve_deepseek_model(model),
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )


class QwenClient(OpenAICompatibleClient):
    """Alibaba Cloud MaaS OpenAI-compatible endpoint for Qwen models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[CacheBackend] = None,
        max_attempts: int = 5,
        request_timeout: float = 60.0,
        base_url: Optional[str] = None,
    ):
        key = api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
        super().__init__(
            base_url=base_url or QWEN_BASE_URL,
            api_key=key,
            cache=cache,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            provider="qwen",
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
        return super().call(
            model=resolve_qwen_model(model),
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )


class DoubaoClient(OpenAICompatibleClient):
    """Volcengine Ark OpenAI-compatible endpoint for Doubao models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[CacheBackend] = None,
        max_attempts: int = 5,
        request_timeout: float = 60.0,
        base_url: Optional[str] = None,
    ):
        key = api_key or os.environ.get("DOUBAO_API_KEY") or os.environ.get("ARK_API_KEY", "")
        super().__init__(
            base_url=base_url or DOUBAO_BASE_URL,
            api_key=key,
            cache=cache,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            provider="doubao",
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
        return super().call(
            model=resolve_doubao_model(model),
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )


class KimiClient(OpenAICompatibleClient):
    """Moonshot (Kimi) OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[CacheBackend] = None,
        max_attempts: int = 5,
        request_timeout: float = 60.0,
        base_url: Optional[str] = None,
    ):
        key = api_key or os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY", "")
        super().__init__(
            base_url=base_url or KIMI_BASE_URL,
            api_key=key,
            cache=cache,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            provider="kimi",
        )

    def _augment_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        model_id = str(payload.get("model", ""))
        out = dict(payload)
        if kimi_omit_temperature(model_id):
            out.pop("temperature", None)
        if kimi_disable_thinking() and kimi_thinking_can_disable(model_id):
            out["extra_body"] = {"thinking": {"type": "disabled"}}
        return out

    def call(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = 800,
    ) -> LLMResponse:
        return super().call(
            model=resolve_kimi_model(model),
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )


class MinimaxClient(OpenAICompatibleClient):
    """MiniMax OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[CacheBackend] = None,
        max_attempts: int = 5,
        request_timeout: float = 60.0,
        base_url: Optional[str] = None,
    ):
        key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        super().__init__(
            base_url=base_url or MINIMAX_BASE_URL,
            api_key=key,
            cache=cache,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            provider="minimax",
        )

    def _augment_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        model_id = str(payload.get("model", ""))
        if minimax_disable_thinking() and minimax_thinking_can_disable(model_id):
            out["extra_body"] = {"thinking": {"type": "disabled"}}
        return out

    def call(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = 800,
    ) -> LLMResponse:
        api_model = resolve_minimax_model(model)
        resp = super().call(
            model=api_model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        cleaned = _strip_thinking_markup(resp.text)
        if cleaned == resp.text:
            return resp
        return LLMResponse(
            text=cleaned,
            model=model,
            cached=resp.cached,
            usage=resp.usage,
            raw=resp.raw,
        )


class RoutingClient(BaseClient):
    """Dispatch each call by model slug to the matching provider client."""

    def __init__(
        self,
        openrouter: Optional[OpenRouterClient] = None,
        deepseek: Optional[DeepSeekClient] = None,
        qwen: Optional[QwenClient] = None,
        doubao: Optional[DoubaoClient] = None,
        kimi: Optional[KimiClient] = None,
        minimax: Optional[MinimaxClient] = None,
        cache: Optional[CacheBackend] = None,
        use_llm_cache: Optional[bool] = None,
    ):
        self._cache = cache if cache is not None else resolve_llm_cache(use_llm_cache)
        self._openrouter = openrouter
        self._deepseek = deepseek
        self._qwen = qwen
        self._doubao = doubao
        self._kimi = kimi
        self._minimax = minimax
        self._lock = threading.Lock()

    def _openrouter_client(self) -> OpenRouterClient:
        if self._openrouter is None:
            with self._lock:
                if self._openrouter is None:
                    self._openrouter = OpenRouterClient(cache=self._cache)
        return self._openrouter

    def _deepseek_client(self) -> DeepSeekClient:
        if self._deepseek is None:
            with self._lock:
                if self._deepseek is None:
                    self._deepseek = DeepSeekClient(cache=self._cache)
        return self._deepseek

    def _qwen_client(self) -> QwenClient:
        if self._qwen is None:
            with self._lock:
                if self._qwen is None:
                    self._qwen = QwenClient(cache=self._cache)
        return self._qwen

    def _doubao_client(self) -> DoubaoClient:
        if self._doubao is None:
            with self._lock:
                if self._doubao is None:
                    self._doubao = DoubaoClient(cache=self._cache)
        return self._doubao

    def _kimi_client(self) -> KimiClient:
        if self._kimi is None:
            with self._lock:
                if self._kimi is None:
                    self._kimi = KimiClient(cache=self._cache)
        return self._kimi

    def _minimax_client(self) -> MinimaxClient:
        if self._minimax is None:
            with self._lock:
                if self._minimax is None:
                    self._minimax = MinimaxClient(cache=self._cache)
        return self._minimax

    def call(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = 800,
    ) -> LLMResponse:
        if is_deepseek_model(model):
            backend = self._deepseek_client()
        elif is_qwen_model(model):
            backend = self._qwen_client()
        elif is_doubao_model(model):
            backend = self._doubao_client()
        elif is_kimi_model(model):
            backend = self._kimi_client()
        elif is_minimax_model(model):
            backend = self._minimax_client()
        else:
            backend = self._openrouter_client()
        resp = backend.call(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            text=resp.text,
            model=model,
            cached=resp.cached,
            usage=resp.usage,
            raw=resp.raw,
        )


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
        if max_tokens <= 32:
            payload = {"choice": choice}
        else:
            payload = {
                "choice": choice,
                "ranking": ranking,
                "scores": {l: round(scores[l], 1) for l in letters},
                "reason": "mock decision (position + content)",
            }
        return LLMResponse(text=_json.dumps(payload, ensure_ascii=False), model=model, cached=False)


def _has_any_api_key() -> bool:
    return bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("DOUBAO_API_KEY")
        or os.environ.get("ARK_API_KEY")
        or os.environ.get("MOONSHOT_API_KEY")
        or os.environ.get("KIMI_API_KEY")
        or os.environ.get("MINIMAX_API_KEY")
    )


def get_client(mock: Optional[bool] = None, use_llm_cache: Optional[bool] = None) -> BaseClient:
    """Return a routing client for all configured providers, or a mock when requested."""
    if mock is True:
        return MockClient()
    has_key = _has_any_api_key()
    if mock is None and not has_key:
        warnings.warn(
            "OPENROUTER_API_KEY / DEEPSEEK_API_KEY / QWEN_API_KEY / DOUBAO_API_KEY / "
            "MOONSHOT_API_KEY / MINIMAX_API_KEY not set; falling back to MockClient."
        )
        return MockClient()
    if mock is False and not has_key:
        raise RuntimeError(
            "mock=False but no OPENROUTER_API_KEY, DEEPSEEK_API_KEY, QWEN_API_KEY, "
            "DOUBAO_API_KEY, MOONSHOT_API_KEY, or MINIMAX_API_KEY set"
        )
    return RoutingClient(use_llm_cache=use_llm_cache)
