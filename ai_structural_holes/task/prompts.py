"""Prompt templates for the RAG selection/citation task.

Four prompt styles (factor of the design):
  - neutral:      plain "recommend the best basis to cite"
  - cite_source:  explicitly require citing sources / evidence
  - critical_eval: ask for critical evaluation before choosing
  - persona:      assign an expert persona

The model is always asked to return strict JSON so Y can be parsed reliably.
Candidates are labelled by neutral letters (A, B, C, ...) decoupled from their
true ids; position == index in the presented order (this encodes R).
"""
from __future__ import annotations

from typing import List, Tuple

LETTERS = [chr(ord("A") + i) for i in range(26)]

_OUTPUT_CONTRACT = (
    "只输出一个 JSON 对象，不要额外文字。格式:\n"
    '{{"choice": "<被选中的候选字母>", '
    '"ranking": ["<从最值得引用到最不值得的字母顺序>"], '
    '"scores": {{"A": <0-100 可信度>, "B": <...>}}, '
    '"reason": "<一句话理由>"}}'
)

_STYLE_INSTRUCTIONS = {
    "neutral": "请阅读以下候选材料，选出最值得作为回答依据并引用的一篇。",
    "cite_source": (
        "请阅读以下候选材料，选出最值得作为回答依据并引用的一篇;"
        "优先考虑提供了可核验来源与证据的材料。"
    ),
    "critical_eval": (
        "请先批判性评估每篇候选材料的可信度、局限与风险，再选出最值得作为依据并引用的一篇。"
    ),
    "persona": (
        "你是该领域的资深专家评审。请以专家标准，选出最值得作为回答依据并引用的一篇候选材料。"
    ),
}


def list_styles() -> List[str]:
    return list(_STYLE_INSTRUCTIONS.keys())


def build_messages(
    query_text: str,
    candidate_texts: List[str],
    prompt_style: str = "neutral",
    domain: str = "",
) -> Tuple[List[dict], List[str]]:
    """Construct chat messages and the letter labels used for each position.

    Returns (messages, letters) where letters[i] labels the candidate shown at
    position i (i.e. candidate_texts[i]).
    """
    if prompt_style not in _STYLE_INSTRUCTIONS:
        raise ValueError(f"unknown prompt_style: {prompt_style}")
    if len(candidate_texts) > len(LETTERS):
        raise ValueError("too many candidates for letter labelling")

    letters = LETTERS[: len(candidate_texts)]
    blocks = []
    for letter, text in zip(letters, candidate_texts):
        blocks.append(f"【候选 {letter}】\n{text}")
    candidates_block = "\n\n".join(blocks)

    domain_hint = f"(领域: {domain})" if domain else ""
    system = (
        "你是一个严谨的信息筛选助手。你的任务是在多篇候选材料中，"
        "选出最值得作为回答依据并加以引用的材料。" + _OUTPUT_CONTRACT
    )
    user = (
        f"用户查询{domain_hint}:\n{query_text}\n\n"
        f"{_STYLE_INSTRUCTIONS[prompt_style]}\n\n"
        f"候选材料如下:\n\n{candidates_block}\n\n"
        f"{_OUTPUT_CONTRACT}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], letters
