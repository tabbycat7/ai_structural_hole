"""Prompt templates for the RAG selection/citation task.

Four prompt styles (factor of the design):
  - neutral:      plain "recommend the best basis to cite"
  - cite_source:  explicitly require citing sources / evidence
  - critical_eval: ask for critical evaluation before choosing
  - persona:      assign an expert persona

The model is asked to return strict JSON so Y can be parsed reliably.
Candidates are labelled by neutral letters (A, B, C, ...) decoupled from their
true ids; position == index in the presented order (this encodes R).

output_mode:
  - minimal (default): {"choice": "<letter>"} only — cheaper, sufficient for Y.
  - full: choice + ranking + scores + reason — for audit / exploratory analysis.
  - cite: {"answer": ..., "cited": [<letters>]} — the real RAG generation task
    (Study 5): the model answers using the candidate passages and marks which
    ones it actually cited (multi-select), so Y = target was cited.
"""
from __future__ import annotations

from typing import List, Tuple

LETTERS = [chr(ord("A") + i) for i in range(26)]

OUTPUT_MODES = ("minimal", "full", "cite")

_OUTPUT_CONTRACT_MINIMAL = (
    '只输出一个 JSON 对象，不要额外文字、不要代码块围栏。格式:\n'
    '{"choice": "<被选中的候选字母>"}'
)

_OUTPUT_CONTRACT_FULL = (
    "只输出一个 JSON 对象，不要额外文字、不要代码块围栏。格式:\n"
    '{"choice": "<被选中的候选字母>", '
    '"ranking": ["<从最值得引用到最不值得的字母顺序>"], '
    '"scores": {"A": <0-100 可信度>, "B": <...>}, '
    '"reason": "<一句话理由>"}'
)

_OUTPUT_CONTRACT_CITE = (
    "只输出一个 JSON 对象，不要额外文字、不要代码块围栏。格式:\n"
    '{"answer": "<简要回答>", '
    '"cited": ["<你在回答中实际引用作为依据的候选字母，可一个或多个>"]}'
)

# Back-compat alias for tests that import the full contract.
_OUTPUT_CONTRACT = _OUTPUT_CONTRACT_FULL

_MAX_TOKENS = {"minimal": 32, "full": 800, "cite": 512}

# Study 6: the model rewrites/expands the user question into search queries
# *before* retrieval. It only sees the question + domain (never any candidate or
# target text), so a query's control/treatment targets still face an identical
# retrieval environment under the same model.
_REWRITE_MAX_TOKENS = 256

_REWRITE_CONTRACT = (
    "只输出一个 JSON 对象，不要额外文字、不要代码块围栏。格式:\n"
    '{"queries": ["<检索式1>", "<检索式2>", ...]}'
)

# In cite mode the task is answering + citing (not single-select), so the style
# instructions are reworded around answering the query.
_CITE_STYLE_INSTRUCTIONS = {
    "neutral": "请基于候选材料回答上述查询。",
    "cite_source": "请基于候选材料回答上述查询;优先使用提供了可核验来源与证据的材料。",
    "critical_eval": "请先批判性评估候选材料的可信度、局限与风险，再基于可靠材料回答上述查询。",
    "persona": "你是该领域的资深专家。请以专家标准，基于候选材料回答上述查询。",
}

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


def output_contract(output_mode: str = "minimal") -> str:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"unknown output_mode: {output_mode}")
    if output_mode == "minimal":
        return _OUTPUT_CONTRACT_MINIMAL
    if output_mode == "cite":
        return _OUTPUT_CONTRACT_CITE
    return _OUTPUT_CONTRACT_FULL


def max_tokens_for_mode(output_mode: str = "minimal") -> int:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"unknown output_mode: {output_mode}")
    return _MAX_TOKENS[output_mode]


def max_tokens_for_rewrite() -> int:
    return _REWRITE_MAX_TOKENS


def build_rewrite_messages(
    query_text: str,
    domain: str = "",
    n_queries: int = 3,
) -> List[dict]:
    """Ask the model to turn a user question into up to `n_queries` search queries.

    The prompt deliberately contains only the user question and domain hint (no
    candidate/target text), so query rewriting is independent of the injected
    target. This keeps the paired (control, treatment) contrast clean while
    letting retrieval vary by model.
    """
    n_queries = max(1, int(n_queries))
    domain_hint = f"(领域: {domain})" if domain else ""
    system = (
        "你是一个检索式改写助手。给定用户问题，你要生成若干条用于全文/向量检索的查询式，"
        "以便更好地从资料库中召回相关材料。只基于问题本身改写，"
        "不要编造具体文档内容或事实，不要回答问题。" + _REWRITE_CONTRACT
    )
    user = (
        f"用户问题{domain_hint}:\n{query_text}\n\n"
        f"请生成不超过 {n_queries} 条检索式(可包含同义词、关键词或更聚焦的表述)。\n"
        f"{_REWRITE_CONTRACT}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_messages(
    query_text: str,
    candidate_texts: List[str],
    prompt_style: str = "neutral",
    domain: str = "",
    output_mode: str = "minimal",
) -> Tuple[List[dict], List[str]]:
    """Construct chat messages and the letter labels used for each position.

    Returns (messages, letters) where letters[i] labels the candidate shown at
    position i (i.e. candidate_texts[i]).
    """
    if prompt_style not in _STYLE_INSTRUCTIONS:
        raise ValueError(f"unknown prompt_style: {prompt_style}")
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"unknown output_mode: {output_mode}")
    if len(candidate_texts) > len(LETTERS):
        raise ValueError("too many candidates for letter labelling")

    contract = output_contract(output_mode)
    letters = LETTERS[: len(candidate_texts)]
    blocks = []
    for letter, text in zip(letters, candidate_texts):
        blocks.append(f"【候选 {letter}】\n{text}")
    candidates_block = "\n\n".join(blocks)

    domain_hint = f"(领域: {domain})" if domain else ""

    if output_mode == "cite":
        system = (
            "你是一个严谨的检索增强问答助手。你将阅读若干候选材料并回答用户问题，"
            "并如实标注你在回答中实际引用了哪些候选材料(可多选)。" + contract
        )
        user = (
            f"用户查询{domain_hint}:\n{query_text}\n\n"
            f"{_CITE_STYLE_INSTRUCTIONS[prompt_style]}\n\n"
            f"候选材料如下:\n\n{candidates_block}\n\n"
            f"请在 cited 中列出你实际引用作为依据的候选字母(可一个或多个)。\n{contract}"
        )
    else:
        system = (
            "你是一个严谨的信息筛选助手。你的任务是在多篇候选材料中，"
            "选出最值得作为回答依据并加以引用的材料。" + contract
        )
        user = (
            f"用户查询{domain_hint}:\n{query_text}\n\n"
            f"{_STYLE_INSTRUCTIONS[prompt_style]}\n\n"
            f"候选材料如下:\n\n{candidates_block}\n\n"
            f"{contract}"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], letters
