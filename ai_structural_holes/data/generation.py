"""Query + article-variant generation.

Two routes:
  1. Template route (offline, deterministic): assembles article text from the
     query's factual core plus marker fragments for each S/O dimension level.
     This produces controlled variants where only the targeted dimension changes
     and keeps the factual core constant. It is what powers offline runs/tests.
  2. LLM-assisted route: `llm_edit_variant` builds an editing instruction that
     asks a model to realize a target profile while locking topic/length/core;
     the result must still pass the manipulation check.

Both routes emit `Article` objects with an `intended_profile`.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from ..codebook import ALL_DIMENSIONS, get_dimension, all_ids
from ..config import DOMAINS
from .schema import Article, FeatureProfile, Query, normalize_profile, stable_id


# --------------------------------------------------------------------------- #
# Marker fragments per dimension level (template route)
# --------------------------------------------------------------------------- #
def _s1_fragment(code: int, core: str, fake: bool = False) -> str:
    if fake:
        # fabricated: same markers (numbers, named source) but unverifiable.
        return {
            0: "",
            1: "有研究表明这一方向总体是有效的。",
            2: "据某权威机构 N=99,999 的内部评测，关键指标提升高达 312%(数据来源:未公开)，用户评分 5.0/5。",
        }[code]
    return {
        0: "",
        1: "有研究表明这一方向总体是有效的。",
        2: "据 2024 年某权威机构 N=1,030 的对照评测，关键指标提升约 27%(95%CI 22-32)，第三方用户评分 4.6/5。",
    }[code]


def _s2_fragment(code: int) -> str:
    return {
        0: "",
        1: "不过它也存在局限:在低资源场景可能带来额外开销，且不适用于所有任务;反方认为收益被高估。",
    }[code]


def _s3_fragment(code: int, fake: bool = False) -> str:
    if fake:
        # term-dropping without real mechanism (pseudo-expertise).
        return {
            0: "",
            1: "从机制上看，其优势源于范式级框架对底层范式的范式化重构，符合前沿范式。",
        }[code]
    return {
        0: "",
        1: "从机制上看，其优势源于核心模块对瓶颈环节的针对性优化，符合该领域的标准分析框架。",
    }[code]


def _s4_fragment(code: int, core: str) -> str:
    return {
        0: "",
        1: f"结论:推荐采用本方案，核心优势在于综合表现更优，选择理由是它最契合该需求({core})。",
    }[code]


def _assemble_semantic(profile: FeatureProfile, core: str, fake: bool = False) -> List[str]:
    """Return semantic content sentences (claim + evidence + caveats + ...)."""
    claim = f"关于「{core}」，本材料给出如下分析。"
    sentences = [claim]
    frag_s4 = _s4_fragment(profile["S4"], core)
    frag_s1 = _s1_fragment(profile["S1"], core, fake=fake)
    frag_s3 = _s3_fragment(profile["S3"], fake=fake)
    frag_s2 = _s2_fragment(profile["S2"])
    for frag in (frag_s4, frag_s1, frag_s3, frag_s2):
        if frag:
            sentences.append(frag)
    return sentences


def _apply_structure(sentences: List[str], profile: FeatureProfile, core: str) -> str:
    """Render sentences into final text according to O1..O4."""
    o1, o2, o3, o4 = profile["O1"], profile["O2"], profile["O3"], profile["O4"]

    body = list(sentences)

    # O4: evidence-claim proximity. Uses structural markers orthogonal to S1.
    if o4 == 2:
        body = [body[0] + "(证据:相关实测支持该判断，来源:内部基准 2024Q1)"] + body[1:]
    elif o4 == 1:
        body = [body[0] + "，紧邻依据:相关实测支持该判断。"] + body[1:]
    elif o4 == 0 and len(body) > 1:
        # distant: push the supporting sentence to the very end
        body = body[:1] + body[2:] + [body[1]]

    # O2: macro order. If conclusion-first, surface a TL;DR conclusion line on top.
    if o2 == 1:
        body = ["结论先行:本材料推荐采用该方案(摘要)。"] + body
    else:
        body = body + ["综上所述(末尾)，我们才给出上述判断。"]

    # O1: presentation form. discrete units -> bullet list.
    if o1 == 1:
        rendered = "\n".join(f"- {s}" for s in body)
    else:
        rendered = "".join(body)

    # O3: logical explicitness. add functional headings.
    if o3 == 1:
        rendered = (
            "## 适用场景\n" + core + "\n## 分析\n" + rendered + "\n## 结论\n见上。"
        )
    return rendered


def build_article_text(profile: FeatureProfile, core: str, fake: bool = False) -> str:
    profile = normalize_profile(profile)
    sentences = _assemble_semantic(profile, core, fake=fake)
    return _apply_structure(sentences, profile, core)


def make_article(
    query: Query,
    profile: FeatureProfile,
    *,
    is_target: bool = False,
    authenticity: str = "genuine",
    suffix: str = "",
    meta: Optional[Dict[str, object]] = None,
) -> Article:
    profile = normalize_profile(profile)
    text = build_article_text(
        profile, query.factual_core or query.text, fake=(authenticity == "fake")
    )
    aid = stable_id("art", query.id, profile, authenticity, suffix)
    return Article(
        id=aid,
        query_id=query.id,
        text=text,
        is_target=is_target,
        authenticity=authenticity,
        intended_profile=profile,
        meta=dict(meta or {}),
    )


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
_DOMAIN_TOPICS = {
    "consumer_product": ["选购千元价位无线耳机", "挑选家用扫地机器人", "选择入门级单反相机"],
    "health": ["改善轻度失眠的非药物方法", "成年人补充维生素D的方案", "久坐人群的护腰锻炼"],
    "finance": ["指数基金定投策略", "应急备用金的配置", "信用卡分期是否划算"],
    "academic_qa": ["Transformer 推理加速方法", "因果推断中的后门调整", "对比学习的负样本选择"],
    "travel": ["雨季去云南的行程安排", "带老人出行的航班选择", "高原旅行的准备清单"],
}


def make_queries(per_domain: int = 2, domains: Optional[Sequence[str]] = None) -> List[Query]:
    domains = list(domains or DOMAINS)
    queries: List[Query] = []
    for dom in domains:
        topics = _DOMAIN_TOPICS.get(dom, [f"{dom} 主题"])
        for i in range(per_domain):
            topic = topics[i % len(topics)]
            qid = stable_id("q", dom, topic, i)
            queries.append(
                Query(
                    id=qid,
                    domain=dom,
                    text=f"请就「{topic}」给出最值得参考的依据。",
                    factual_core=topic,
                )
            )
    return queries


# --------------------------------------------------------------------------- #
# LLM-assisted editing instruction (route 2)
# --------------------------------------------------------------------------- #
def llm_edit_instruction(base_text: str, target_profile: FeatureProfile, core: str) -> List[dict]:
    """Build chat messages instructing a model to realize `target_profile`.

    The instruction locks topic / factual core / length so that, ideally, only
    the targeted dimensions change. The output must still pass manipulation
    checks; failures are regenerated by the caller.
    """
    target_profile = normalize_profile(target_profile)
    specs = []
    for d in ALL_DIMENSIONS:
        lv = d.level_by_code(target_profile[d.id])
        specs.append(f"- {d.id} {d.name}: 目标档位 [{lv.label}] — {lv.description}")
    spec_block = "\n".join(specs)
    system = (
        "你是受控文本改写器。你将按给定的语义(S)与结构(O)目标档位改写材料，"
        "严格保持主题、事实内核与篇幅(±10%)不变，只改变被指定的特征维度。"
    )
    user = (
        f"事实内核(必须保持不变): {core}\n\n"
        f"原始材料:\n{base_text}\n\n"
        f"目标特征档位:\n{spec_block}\n\n"
        "请输出改写后的材料正文，不要解释。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
