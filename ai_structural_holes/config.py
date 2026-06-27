"""Global configuration and paths.

Reads optional overrides from environment / .env. No secrets are hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency; safe if missing
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(var: str, default: str) -> Path:
    raw = os.environ.get(var, default)
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    cache_dir: Path = field(default_factory=lambda: _path_from_env("ASH_CACHE_DIR", ".cache/llm"))
    output_dir: Path = field(default_factory=lambda: _path_from_env("ASH_OUTPUT_DIR", "outputs"))
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    def ensure(self) -> "Paths":
        for p in (self.cache_dir, self.output_dir, self.data_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()


# Default model roster (all routed through OpenRouter). Override per-run as needed.
DEFAULT_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-pro-1.5",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.1-70b-instruct",
]

# Task domains, stratified by verifiability / stakes (part of Q).
DOMAINS = ["consumer_product", "health", "finance", "academic_qa", "travel"]

# Prompt styles (how the selection task is framed).
PROMPT_STYLES = ["neutral", "cite_source", "critical_eval", "persona"]

# Competition environment R: candidate-set sizes to test.
CANDIDATE_SET_SIZES = [3, 5, 8]
