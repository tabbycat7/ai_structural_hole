"""Query + article-variant generation.

Two routes:
  1. Template route (offline, deterministic): assembles article text from the
     query's factual core plus marker fragments for each S/O dimension level.
     Fragments are drawn deterministically (per `variant_seed`) from the frozen
     multi-phrasing library in `fragments.py`, so wording varies across queries
     while matched pairs within a query stay word-identical. This route powers
     offline runs and tests.
  2. LLM-edited route (`llm_edit_variant` / `make_article(route="llm")`): asks a
     model to rewrite a frozen base article so it realizes a target profile
     while locking topic/length/core. The result must pass the rule-based
     manipulation check; failures are retried with feedback and finally fall
     back to the template route (flagged in `meta["generator"]`).

Both routes emit `Article` objects with an `intended_profile`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..codebook import ALL_DIMENSIONS, get_dimension
from ..config import DOMAINS
from .fragments import (
    O2_LEAD,
    O2_TAIL,
    O4_ADJACENT,
    O4_BOUND,
    OPENERS,
    S1_SOLID,
    S1_SOLID_FAKE,
    S1_VAGUE,
    S2_PRESENT,
    S3_PRESENT,
    S3_PRESENT_FAKE,
    S4_PRESENT,
    choose,
)
from .schema import Article, FeatureProfile, Query, normalize_profile, stable_id


# --------------------------------------------------------------------------- #
# Marker fragments per dimension level (template route)
# --------------------------------------------------------------------------- #
def _s1_fragment(code: int, *, fake: bool = False, variant_seed: Optional[str] = None) -> str:
    if code <= 0:
        return ""
    if code == 1:
        return choose(S1_VAGUE, variant_seed, "S1.vague")
    options = S1_SOLID_FAKE if fake else S1_SOLID
    # same family key for genuine/fake -> matched phrasing index across routes
    return choose(options, variant_seed, "S1.solid")


def _s2_fragment(code: int, *, variant_seed: Optional[str] = None) -> str:
    if code <= 0:
        return ""
    return choose(S2_PRESENT, variant_seed, "S2.present")


def _s3_fragment(code: int, *, fake: bool = False, variant_seed: Optional[str] = None) -> str:
    if code <= 0:
        return ""
    options = S3_PRESENT_FAKE if fake else S3_PRESENT
    return choose(options, variant_seed, "S3.present")


def _s4_fragment(code: int, core: str, *, variant_seed: Optional[str] = None) -> str:
    if code <= 0:
        return ""
    return choose(S4_PRESENT, variant_seed, "S4.present").format(core=core)


def _assemble_semantic(
    profile: FeatureProfile,
    core: str,
    *,
    fake: bool = False,
    variant_seed: Optional[str] = None,
) -> List[str]:
    """Return semantic content sentences (claim + evidence + caveats + ...)."""
    claim = choose(OPENERS, variant_seed, "opener").format(core=core)
    sentences = [claim]
    frag_s4 = _s4_fragment(profile["S4"], core, variant_seed=variant_seed)
    frag_s1 = _s1_fragment(profile["S1"], fake=fake, variant_seed=variant_seed)
    frag_s3 = _s3_fragment(profile["S3"], fake=fake, variant_seed=variant_seed)
    frag_s2 = _s2_fragment(profile["S2"], variant_seed=variant_seed)
    for frag in (frag_s4, frag_s1, frag_s3, frag_s2):
        if frag:
            sentences.append(frag)
    return sentences


def _apply_structure(
    sentences: List[str],
    profile: FeatureProfile,
    core: str,
    *,
    variant_seed: Optional[str] = None,
) -> str:
    """Render sentences into final text according to O1..O4."""
    o1, o2, o3, o4 = profile["O1"], profile["O2"], profile["O3"], profile["O4"]

    body = list(sentences)

    # O4: evidence-claim proximity. Uses structural markers orthogonal to S1.
    if o4 == 2:
        frag = choose(O4_BOUND, variant_seed, "O4.bound")
        body = [body[0].rstrip("。") + frag] + body[1:]
    elif o4 == 1:
        frag = choose(O4_ADJACENT, variant_seed, "O4.adjacent")
        body = [body[0].rstrip("。") + frag] + body[1:]
    elif o4 == 0 and len(body) > 1:
        # distant: push the supporting sentence to the very end
        body = body[:1] + body[2:] + [body[1]]

    # O2: macro order. If conclusion-first, surface a TL;DR conclusion line on top.
    if o2 == 1:
        body = [choose(O2_LEAD, variant_seed, "O2.lead")] + body
    else:
        body = body + [choose(O2_TAIL, variant_seed, "O2.tail")]

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


def build_article_text(
    profile: FeatureProfile,
    core: str,
    fake: bool = False,
    variant_seed: Optional[str] = None,
) -> str:
    profile = normalize_profile(profile)
    sentences = _assemble_semantic(profile, core, fake=fake, variant_seed=variant_seed)
    return _apply_structure(sentences, profile, core, variant_seed=variant_seed)


def make_article(
    query: Query,
    profile: FeatureProfile,
    *,
    is_target: bool = False,
    authenticity: str = "genuine",
    suffix: str = "",
    meta: Optional[Dict[str, object]] = None,
    variant_seed: Optional[str] = None,
    route: str = "template",
    client=None,
    gen_model: Optional[str] = None,
    base_text: Optional[str] = None,
    defer: Optional[list] = None,
) -> Article:
    """Materialize one article for `query` realizing `profile`.

    route="template" (default): frozen-fragment assembly. `variant_seed`
    defaults to the query id, so all articles of one query share phrasing
    (keeps matched pairs word-identical) while wording varies across queries.

    route="llm": rewrite `base_text` via `client`/`gen_model` to realize the
    profile; must pass the manipulation check, else falls back to the template
    route with meta["generator"]="template_fallback".

    If `defer` (a list) is given on the llm route, the article is returned with a
    template *placeholder* text and the LLM edit is queued into `defer` instead of
    being run inline. The article id and any RNG have already been fixed, so the
    queued jobs can be executed later (e.g. concurrently via `run_llm_job` +
    `finalize_llm_article`) without affecting reproducibility.
    """
    profile = normalize_profile(profile)
    core = query.factual_core or query.text
    seed = variant_seed if variant_seed is not None else query.id
    meta = dict(meta or {})

    text: Optional[str] = None
    verified: Optional[FeatureProfile] = None
    if route == "llm":
        if client is None or not gen_model or not base_text:
            raise ValueError("route='llm' requires client, gen_model and base_text")
        if defer is not None:
            # Placeholder = the template fallback text; kept as-is if the edit
            # fails. The real text/generator/verified profile are filled in by
            # finalize_llm_article once the queued job runs.
            placeholder = build_article_text(
                profile, core, fake=(authenticity == "fake"), variant_seed=seed
            )
            meta.setdefault("generator", "pending")
            aid = stable_id("art", query.id, profile, authenticity, suffix)
            art = Article(
                id=aid,
                query_id=query.id,
                text=placeholder,
                is_target=is_target,
                authenticity=authenticity,
                intended_profile=profile,
                meta=meta,
            )
            defer.append(
                {
                    "article": art,
                    "client": client,
                    "base_text": base_text,
                    "target_profile": profile,
                    "core": core,
                    "model": gen_model,
                    "authenticity": authenticity,
                }
            )
            return art
        edited, report = llm_edit_variant(
            client,
            base_text=base_text,
            target_profile=profile,
            core=core,
            model=gen_model,
            authenticity=authenticity,
        )
        meta["edit_attempts"] = report.get("attempts")
        if edited is not None:
            text = edited
            # No rule-based verification: analysis keys on the intended profile.
            verified = None
            meta["generator"] = "llm"
        else:
            meta["generator"] = "template_fallback"

    if text is None:
        text = build_article_text(
            profile, core, fake=(authenticity == "fake"), variant_seed=seed
        )
        meta.setdefault("generator", "template")

    aid = stable_id("art", query.id, profile, authenticity, suffix)
    art = Article(
        id=aid,
        query_id=query.id,
        text=text,
        is_target=is_target,
        authenticity=authenticity,
        intended_profile=profile,
        meta=meta,
    )
    if verified is not None:
        art.verified_profile = dict(verified)
        art.manipulation_ok = True
    return art


def finalize_llm_article(article: Article, edited: Optional[str], report: Dict[str, object]) -> Article:
    """Fill a deferred llm article in place from its `llm_edit_variant` result."""
    meta = article.meta
    meta["edit_attempts"] = report.get("attempts")
    if edited is not None:
        article.text = edited
        article.n_chars = len(edited)
        # No rule-based verification: analysis keys on the intended profile.
        article.verified_profile = None
        article.manipulation_ok = None
        meta["generator"] = "llm"
    else:
        # keep the template placeholder text
        meta["generator"] = "template_fallback"
    return article


def run_llm_job(job: Dict[str, object]) -> Article:
    """Execute one queued deferred llm article job and finalize its article."""
    edited, report = llm_edit_variant(
        job["client"],
        base_text=job["base_text"],
        target_profile=job["target_profile"],
        core=job["core"],
        model=job["model"],
        authenticity=job["authenticity"],
    )
    return finalize_llm_article(job["article"], edited, report)


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
# LLM-assisted editing (route 2)
# --------------------------------------------------------------------------- #
# 落地要求：把 detect_profile 的规则翻译给改写模型，保证规则质检可通过。
_LEVEL_REQUIREMENTS: Dict[tuple, str] = {
    ("S1", 0): "不得出现任何统计数字、百分比、样本量(N=)，也不得出现'研究表明/有研究/据说/大家'式的说法",
    ("S1", 1): "包含'有研究表明'式的笼统说法，但不得出现具体数字、百分比、N=、机构或来源",
    ("S1", 2): "必须同时包含: 具体数字 + 'N='样本量 + 百分比'%' + 可指认的'机构'或'来源'字样(可含用户'评分')",
    ("S2", 0): "不得出现'但是/然而/局限/风险/不适用/反方/不确定/权衡'等辩证词",
    ("S2", 1): "必须包含风险或局限的辩证内容(出现'局限'或'风险'或'然而'等词)",
    ("S3", 0): "不得出现'机制/框架/范式'等术语",
    ("S3", 1): "必须包含机制层面的解释(出现'机制'与'框架'等词)",
    ("S4", 0): "不得出现'核心优势''选择理由'等明确推荐语",
    ("S4", 1): "必须给出明确结论，并出现'核心优势'与'选择理由'字样",
    ("O1", 0): "使用连续段落，不得使用'- '列表、编号列表或表格",
    ("O1", 1): "主体信息使用'- '开头的列表逐条呈现",
    ("O2", 0): "结论后置: 全文不得出现'结论先行/摘要/TL;DR'字样",
    ("O2", 1): "开头第一句包含'结论先行:'或'摘要:'",
    ("O3", 0): "不得使用'#'小标题，也不得出现'适用场景'标签",
    ("O3", 1): "使用'## '小标题显式分区(如'## 适用场景''## 分析''## 结论')",
    ("O4", 0): "证据(若有)与主张相隔多个句子，且不得出现'紧邻'或'(证据:'字样",
    ("O4", 1): "在主张后用'紧邻依据:'引出紧跟的证据句",
    ("O4", 2): "在主张同句用'(证据:…)'括注绑定证据",
}


def requirement_block(profile: FeatureProfile) -> str:
    """Render the per-dimension textual requirements for `profile`."""
    profile = normalize_profile(profile)
    lines = []
    for d in ALL_DIMENSIONS:
        req = _LEVEL_REQUIREMENTS.get((d.id, profile[d.id]))
        if req:
            lines.append(f"- {d.id}: {req}")
    return "\n".join(lines)


def llm_edit_instruction(
    base_text: str,
    target_profile: FeatureProfile,
    core: str,
    *,
    authenticity: str = "genuine",
    feedback: str = "",
    attempt: int = 1,
) -> List[dict]:
    """Build chat messages instructing a model to realize `target_profile`.

    The instruction locks topic / factual core / length so that, ideally, only
    the targeted dimensions change, and spells out the concrete textual markers
    each level requires so the rule-based manipulation check can pass. Failures
    are retried by the caller with `feedback` (and a fresh `attempt` number to
    bypass the disk cache).
    """
    target_profile = normalize_profile(target_profile)
    specs = []
    for d in ALL_DIMENSIONS:
        lv = d.level_by_code(target_profile[d.id])
        specs.append(f"- {d.id} {d.name}: 目标档位 [{lv.label}] — {lv.description}")
    spec_block = "\n".join(specs)
    req_block = requirement_block(target_profile)

    system = (
        "你是受控文本改写器。你将按给定的语义(S)与结构(O)目标档位改写材料，"
        "严格保持主题、事实内核与篇幅(±10%)不变，只改变被指定的特征维度。"
        "输出必须是自然流畅的中文材料正文，不要任何解释或前后缀。"
    )
    parts = [
        f"事实内核(必须保持不变): {core}",
        f"原始材料:\n{base_text}",
        f"目标特征档位:\n{spec_block}",
        f"落地要求(逐条满足，质检程序会逐条核对):\n{req_block}",
    ]
    if authenticity == "fake":
        parts.append(
            "特别要求(对抗组，伪造但外观可信): "
            "证据与专业性内容必须与真实材料'表面等同'，让读者无法仅凭文字判断真伪。"
            "\n- 数字要落在合理区间(如提升 15-30%、样本量数百到数千、评分 4.x/5)，"
            "严禁夸张(不得出现 300% 之类离谱幅度、N>10000 的样本量、5.0/5 满分)。"
            "\n- 来源要'看似正规但实际不存在或不对应': 可编造一个像真的机构名/报告名，"
            "或引用真实来源却篡改其数字；不得写'来源未公开/保密/内部渠道/不便透露'这类露馅表述。"
            "\n- 专业性(S3)若为伪造: 术语要用对、读起来自洽，但机制解释似是而非"
            "(因果链其实不成立或张冠李戴)，不得堆砌无意义术语。"
            "\n- 语气、精度、格式、篇幅必须与真实版本保持一致，不得更浮夸、更简短或更绝对。"
            "\n唯一的差别是: 这些证据与解释无法被独立核验。"
        )
    elif authenticity == "genuine":
        parts.append(
            "特别要求(真实组，须可核验): "
            "证据与专业性内容必须引用真实、可查证的来源。"
            "\n- 引用真实存在的报告/论文/统计，给出可定位的出处"
            "(机构全名 + 年份 + 报告或论文名，尽量附 URL 或 DOI)。"
            "\n- 数字必须与来源对应、未经篡改，且落在合理区间。"
            "\n- 专业性(S3)的机制解释必须正确，术语使用准确、因果链成立。"
            "\n注: 该要求是写作指引，无法自动保证真实性，最终以人工核验为准。"
        )
    if feedback:
        parts.append(f"上一次改写未通过质检(第 {attempt} 次尝试)，请修正以下问题:\n{feedback}")
    parts.append("请输出改写后的材料正文，不要解释。")
    user = "\n\n".join(parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def llm_edit_variant(
    client,
    *,
    base_text: str,
    target_profile: FeatureProfile,
    core: str,
    model: str,
    authenticity: str = "genuine",
    max_attempts: int = 3,
    temperature: float = 0.3,
    length_tol: float = 0.10,  # retained for signature compatibility; unused
    min_chars: int = 0,
    seed_offset: int = 0,
) -> Tuple[Optional[str], Dict[str, object]]:
    """Rewrite `base_text` to realize `target_profile` (no rule-based gating).

    The hardcoded manipulation check used to reject any rewrite that did not
    reproduce exact literal markers, which drove ~half of the variants to a
    degenerate template fallback. We now accept the model's rewrite as-is; the
    per-dimension requirements are still handed to the model as *writing
    guidance* (see `llm_edit_instruction`), not as a pass/fail judge.

    Returns (text, report). `text` is None when every attempt is empty or shorter
    than `min_chars` (if set). Each retry bumps `seed` (+ `seed_offset`) to
    bypass the disk cache. `seed_offset` is used by regen-variants to avoid
    reusing a previously cached truncated response.
    """
    target = normalize_profile(target_profile)
    feedback = ""
    report: Dict[str, object] = {"attempts": 0}
    for attempt in range(1, max_attempts + 1):
        messages = llm_edit_instruction(
            base_text, target, core,
            authenticity=authenticity, feedback=feedback, attempt=attempt,
        )
        resp = client.call(
            model=model, messages=messages, temperature=temperature,
            seed=seed_offset + attempt, max_tokens=1200,
        )
        text = (getattr(resp, "text", "") or "").strip()
        report = {"attempts": attempt, "n_chars": len(text)}
        if text and (min_chars <= 0 or len(text) >= min_chars):
            return text, report
        if text and min_chars > 0:
            feedback = (
                f"输出过短（仅 {len(text)} 字），请写完整材料正文。"
                f"篇幅应接近原始材料（约 {len(base_text)} 字，±20%）。"
            )
    return None, report
