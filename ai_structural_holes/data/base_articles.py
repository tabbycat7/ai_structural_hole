"""Frozen LLM-generated baseline articles (the controlled-experiment substrate).

A *base article* is an all-baseline (every S/O dimension at code 0) plain-prose
piece about one query's factual core. It is generated once by an LLM, validated
with the rule-based detector, frozen to `data/base_articles/<query_id>.json`
(commit it!), and then reused as the common origin from which all LLM-edited
variants are rewritten. Freezing guarantees that the control condition stays
byte-identical for the whole experiment and stays human-auditable.

If generation cannot produce a valid all-baseline text within `max_attempts`,
the record falls back to the template route (flagged `generator=
"template_fallback"`) so the pipeline still runs offline / with the mock client.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..codebook import baseline_profile, get_dimension
from ..config import PATHS
from .generation import build_article_text, requirement_block
from .manipulation_check import detect_profile
from .schema import Query

MIN_CHARS = 100
TARGET_CHARS = (180, 320)


def base_dir(root: Optional[Path] = None) -> Path:
    d = (root or PATHS.data_dir) / "base_articles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def base_article_messages(query: Query, feedback: str = "", attempt: int = 1) -> List[dict]:
    """Chat messages asking for a plain, all-baseline article about the core."""
    core = query.factual_core or query.text
    system = (
        "你是资料撰写者。你将就给定主题写一段中立、朴素的说明性材料，"
        "供后续受控改写实验作为底稿。只输出材料正文，不要任何解释。"
    )
    parts = [
        f"主题(事实内核): {core}",
        f"篇幅: 约 {TARGET_CHARS[0]}~{TARGET_CHARS[1]} 个汉字，单段或少数几段连续散文。",
        "写作约束(逐条满足，质检程序会逐条核对):\n" + requirement_block(baseline_profile()),
        "整体基调: 平实描述该主题的一般情况，观点含糊、不下明确结论，收尾停留在泛泛的总结。",
    ]
    if feedback:
        parts.append(f"上一稿未通过质检(第 {attempt} 次尝试)，请修正以下问题:\n{feedback}")
    parts.append("请输出材料正文。")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def validate_base_text(text: str) -> Dict[str, object]:
    """Check that `text` is a plausible all-baseline article."""
    problems: List[str] = []
    text = (text or "").strip()
    if len(text) < MIN_CHARS:
        problems.append(f"正文过短(<{MIN_CHARS} 字)，请写约 {TARGET_CHARS[0]}~{TARGET_CHARS[1]} 字。")
    if text.startswith("{") or text.startswith("["):
        problems.append("输出疑似 JSON/列表结构，需要的是自然语言正文。")
    detected = detect_profile(text)
    for dim_id, code in detected.items():
        if code != 0:
            lv = get_dimension(dim_id).level_by_code(code)
            problems.append(f"{dim_id} 被检测为非基线档位 [{lv.label}]，请移除相应内容。")
    return {"ok": not problems, "detected": detected, "problems": problems}


def generate_base_article(
    client,
    query: Query,
    model: str,
    *,
    max_attempts: int = 5,
    temperature: float = 0.7,
) -> Dict[str, object]:
    """Generate + validate one base article; fall back to template on failure."""
    core = query.factual_core or query.text
    feedback = ""
    text, generator, attempts = "", "llm", 0
    detected: Dict[str, int] = {}

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        messages = base_article_messages(query, feedback=feedback, attempt=attempt)
        resp = client.call(
            model=model, messages=messages, temperature=temperature,
            seed=attempt, max_tokens=1200,
        )
        candidate = (getattr(resp, "text", "") or "").strip()
        result = validate_base_text(candidate)
        detected = result["detected"]
        if result["ok"]:
            text = candidate
            break
        feedback = "\n".join(f"- {p}" for p in result["problems"])
    else:
        # all attempts failed -> deterministic template baseline keeps things running
        text = build_article_text(baseline_profile(), core, variant_seed=query.id)
        generator = "template_fallback"
        detected = detect_profile(text)

    return {
        "query_id": query.id,
        "domain": query.domain,
        "query_text": query.text,
        "factual_core": core,
        "text": text,
        "model": model,
        "generator": generator,
        "attempts": attempts,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detected_profile": detected,
        "n_chars": len(text),
        "validated": all(v == 0 for v in detected.values()),
    }


def save_base_article(record: Dict[str, object], root: Optional[Path] = None) -> Path:
    path = base_dir(root) / f"{record['query_id']}.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_base_articles(
    root: Optional[Path] = None, validated_only: bool = True
) -> Dict[str, Dict[str, object]]:
    """Load frozen base-article records keyed by query_id."""
    out: Dict[str, Dict[str, object]] = {}
    d = base_dir(root)
    for path in sorted(d.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if validated_only and not record.get("validated"):
            continue
        out[record["query_id"]] = record
    return out


def load_base_texts(root: Optional[Path] = None) -> Dict[str, str]:
    """query_id -> frozen base text (validated records only)."""
    return {qid: r["text"] for qid, r in load_base_articles(root).items()}
