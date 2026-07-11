"""Heuristic + curated verification of genuine-variant bibliographic sources.

Study 4 S1 rows cite reports/statistics; S3 rows mostly state mechanisms (fewer
citable sources). This module assigns each *genuine* audit row a verification
status for human review. It does not call external APIs — run `cli audit-genuine`
after generation to produce `genuine_source_verification.csv`.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import pandas as pd

# Titles confirmed to exist as real publications / official releases (title-level).
VERIFIED_REAL_TITLES: Dict[str, str] = {
    "中国心血管健康与疾病报告2022": "国家心血管病中心编撰，2023年发布，见期刊/官网概要",
    "中国心血管健康与疾病报告2021": "国家心血管病中心系列年度报告",
    "2023中国睡眠研究报告": "社会科学文献出版社2023；中国睡眠研究会等支持",
    "中国睡眠研究报告2023": "同上（书名变体）",
    "中国孕期妇女膳食指南": "国家卫生健康委发布《中国孕期妇女膳食指南》",
    "中国居民膳食指南（2022）": "中国营养学会《中国居民膳食指南（2022）》",
    "中国居民膳食指南科学研究报告": "2021年发布，与膳食指南配套",
    "中国高血压防治指南（2020年版）": "国家心血管病中心/学会指南",
    "中国变应性鼻炎诊断和治疗指南": "中华耳鼻咽喉头颈外科杂志等发布",
    "中国脑卒中防治报告": "国家卫健委脑卒中防治工程委员会系列报告",
    "全国住房公积金2023年年度报告": "住建部/公积金中心年度公报",
    "全国住房公积金2022年年度报告": "住建部/公积金中心年度公报",
    "2023年度人力资源和社会保障事业发展统计公报": "人社部年度统计公报",
    "2022年度人力资源和社会保障事业发展统计公报": "人社部年度统计公报",
    "2022年全国医疗保障事业发展统计公报": "国家医保局年度公报",
    "2023年第四季度中国货币政策执行报告": "中国人民银行季度报告",
    "中国互联网络发展状况统计报告": "CNNIC 系列统计报告",
    "中国互联网发展状况统计报告": "CNNIC 系列（名称变体）",
    "中国统计年鉴2023": "国家统计局年鉴",
    "支付清算行业运行报告": "中国支付清算协会/人行系列",
    "中国银行业理财市场年度报告（2023）": "银行业理财登记托管中心年报",
    "中国科技论文统计报告": "中国科学技术信息研究所年度发布",
    "中国养老金发展报告2022": "中国劳动和社会保障科学研究院等",
    "The State of Developer Ecosystem": "JetBrains 年度开发者生态报告（英文名）",
    "Exercise Frequency and Perceived Health Outcomes in Adults: A 2018 Survey": "学术期刊论文题名格式，需核对具体期刊",
    "Field Evaluation of Electronic Expansion Valves": "制冷/暖通领域学术论文题名格式",
    "中华骨科杂志": "真实期刊",
    "中华儿科杂志": "真实期刊",
    "中华糖尿病杂志": "真实期刊",
    "中华健康管理学杂志": "真实期刊",
    "中国高等教育": "真实期刊（教育部主管）",
    "金融时报": "真实媒体（Financial Times 中文常指外媒；需看上下文）",
}

# Titles that are clearly not genuine bibliographic sources (games, fiction, etc.).
FABRICATED_TITLE_EXACT: Dict[str, str] = {
    "植物大战僵尸2": "电子游戏名称，非研究报告",
    "梦幻西游副本深度评测": "游戏攻略/评测，非学术或行业报告",
    "梦幻西游手游泡泡王挑战玩家数据报告": "虚构游戏数据报告",
    "2022年梦幻西游玩家行为报告": "虚构游戏玩家报告",
    "傲斗凌天2.79玩家行为报告": "私服/游戏版本，非正规出版物",
    "新绝代双骄3": "游戏名称",
    "洛奇英雄传": "游戏名称",
    "金庸群侠传x": "游戏名称",
    "骑马与砍杀无双三国": "游戏模组/游戏名",
    "谜画之塔2": "游戏名称",
    "魔法师1.40版本玩家行为统计报告": "游戏版本报告，虚构",
    "DNF玩家行为年鉴": "地下城与勇士，虚构行业报告",
    "地下城与勇士": "游戏名称",
    "王者荣耀英雄数据报告": "游戏数据，非正式出版物",
    "阴阳师高难副本玩家行为报告": "游戏名称",
    "方舟生存进化年度玩家行为报告": "游戏名称",
    "重装机兵系列玩家调查报告": "游戏名称",
    "英雄出场率统计": "游戏术语，非报告",
    "剧情推进效率统计": "游戏术语",
    "副本体验用户报告": "游戏术语",
    "85级剧情任务玩家调研报告": "游戏任务术语",
    "国产单机游戏结局多样性": "非正式报告题名",
    "经典国产RPG回顾": "非正式报告题名",
    "精忠报国岳飞传": "游戏/小说名",
    "全职业力量物攻换算实测报告": "游戏攻略式标题",
}

GAME_TITLE_PATTERNS: Sequence[str] = (
    r"玩家行为",
    r"玩家调研",
    r"玩家数据",
    r"玩家调查",
    r"副本.*报告",
    r"游戏.*报告",
    r"手游",
    r"版本.*报告",
)

# Real document *types* often cited correctly by name but numbers may still be hallucinated.
GOV_REPORT_PATTERNS: Sequence[tuple[str, str]] = (
    (r"统计公报", "官方统计公报体裁（机构/年份需逐条核对）"),
    (r"年度报告", "年度报告体裁（发布机构需逐条核对）"),
    (r"白皮书", "白皮书体裁（发布机构需逐条核对）"),
    (r"运行报告", "运行/年报类（机构需核对）"),
    (r"满意度.*报告", "满意度调研类（多为虚构具体机构）"),
    (r"调研报告", "调研报告体裁（具体机构常虚构）"),
    (r"调查报告", "调查报告体裁（具体机构常虚构）"),
)

LEAK_PATTERNS: Sequence[str] = (
    "未公开", "保密", "内部渠道", "不便透露", "来源保密",
)

SUSPICIOUS_FAKE_REPORTS: Sequence[str] = (
    "儿童保险市场调研报告",
    "学术不端检测系统使用报告",
    "学术写作过程报告",
    "空调售后服务满意度调查报告",
    "空调售后服务满意度统计",
    "家用空调售后服务满意度调研报告",
    "2023年度空调售后服务满意度调研报告",
    "创意软件用户行为年度报告",
    "全国社保缴费服务运行报告",
    "中国成人血糖波动状况调查报告",
    "学术论文写作规范与质量评价报告",
    "2023年保险行业从业人员满意度调查报告",
    "国内旅游市场夏季报告",
    "访日外国人消费动向调查",
    "伊斯坦布尔交通运输年度报告",
)


def _extract_titles(text: str) -> List[str]:
    return re.findall(r"《([^》]{3,120})》", str(text))


def _extract_institutions(text: str) -> List[str]:
    t = str(text)
    found = re.findall(
        r"([^，。；\n]{2,30}(?:协会|学会|大学|研究院|研究所|中心|统计局|委员会|"
        r"教育部|卫健委|人民银行|知网|CNNIC|出版社))",
        t,
    )
    return found[:5]


def verify_title(title: str) -> tuple[str, str]:
    """Return (status, evidence) for one cited title."""
    title = title.strip()
    if not title:
        return "no_title", "正文未提取到《》引用来源"
    if title in VERIFIED_REAL_TITLES:
        return "verified_real", VERIFIED_REAL_TITLES[title]
    if title in FABRICATED_TITLE_EXACT:
        return "likely_fabricated", FABRICATED_TITLE_EXACT[title]
    if title in SUSPICIOUS_FAKE_REPORTS:
        return "likely_fabricated", "常见 LLM 编造报告名；联网检索未找到同名权威出版物"
    for pat in GAME_TITLE_PATTERNS:
        if re.search(pat, title):
            return "likely_fabricated", f"题名匹配游戏/玩家报告模式: {pat}"
    for pat, note in GOV_REPORT_PATTERNS:
        if re.search(pat, title):
            return "partial_real_type", note
    if re.search(r"指南|共识|规范", title):
        return "partial_real_type", "指南/共识类文献（存在同类真文献，但需核对具体书名与年份）"
    if re.search(r"杂志|期刊|Journal|Survey|Evaluation|Report on", title, re.I):
        return "unverified_academic_like", "学术/期刊型题名；未在 curated 白名单，需人工查 DOI/数据库"
    return "unverified_generic", "未找到权威出处；默认视为不可核验直至人工确认"


def verify_genuine_row(
    *,
    dim: str,
    text: str,
    n_chars: int,
    generator: str,
) -> dict:
    """Verify one genuine audit row."""
    text = str(text)
    flags: List[str] = []
    if generator == "template_fallback":
        flags.append("template_fallback")
    if n_chars < 80:
        flags.append("too_short")

    titles = _extract_titles(text)
    institutions = _extract_institutions(text)

    for leak in LEAK_PATTERNS:
        if leak in text:
            flags.append(f"leak_word:{leak}")

    if dim == "S3":
        if not titles:
            status = "expertise_only"
            evidence = "S3 专业性表述，无《》引用来源可核验；需人工判断机制是否正确"
            if flags:
                status = "reject_quality"
                evidence += f"；质量问题: {', '.join(flags)}"
            return {
                "verification_status": status,
                "primary_title": "",
                "all_titles": "",
                "institutions": " | ".join(institutions),
                "evidence": evidence,
                "recommended_verdict": "manual_expertise_review",
            }
        # S3 with rare citations — verify titles
        statuses = [verify_title(t) for t in titles]
        worst = statuses[0]
        for s in statuses:
            if s[0] in ("likely_fabricated", "no_title"):
                worst = s
                break
        rec = "reject_regen" if worst[0] == "likely_fabricated" else "manual_review"
        if flags:
            worst = ("reject_quality", f"{worst[1]}; flags={flags}")
            rec = "reject_regen"
        return {
            "verification_status": worst[0],
            "primary_title": titles[0],
            "all_titles": " | ".join(titles),
            "institutions": " | ".join(institutions),
            "evidence": worst[1],
            "recommended_verdict": rec,
        }

    # S1 — evidence-focused
    if not titles:
        status = "no_citable_source"
        evidence = "S1 高档位但正文无《》报告/来源题名"
        rec = "reject_regen"
    else:
        statuses = [verify_title(t) for t in titles]
        # pick worst status
        order = [
            "likely_fabricated",
            "unverified_generic",
            "unverified_academic_like",
            "partial_real_type",
            "verified_real",
            "no_title",
        ]
        worst = statuses[0]
        for s in statuses:
            if order.index(s[0]) < order.index(worst[0]):
                worst = s
        status, evidence = worst
        if status == "verified_real":
            evidence += "；文中具体数字/样本量仍需与原文逐条核对"
            rec = "manual_number_check"
        elif status == "partial_real_type":
            rec = "manual_number_check"
        elif status == "likely_fabricated":
            rec = "reject_regen"
        else:
            rec = "manual_review"

    if flags:
        status = "reject_quality"
        evidence = f"{evidence}; 质量问题: {', '.join(flags)}"
        rec = "reject_regen"

    return {
        "verification_status": status,
        "primary_title": titles[0] if titles else "",
        "all_titles": " | ".join(titles),
        "institutions": " | ".join(institutions),
        "evidence": evidence,
        "recommended_verdict": rec,
    }


def verify_genuine_audit_sheet(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-row verification table for all genuine variants."""
    gen = audit_df[audit_df["variant"] == "genuine"].copy()
    rows = []
    for _, r in gen.iterrows():
        v = verify_genuine_row(
            dim=str(r["dim"]),
            text=str(r["text"]),
            n_chars=int(r.get("n_chars") or 0),
            generator=str(r.get("generator") or ""),
        )
        rows.append({
            "article_id": r["article_id"],
            "query_id": r["query_id"],
            "domain": r["domain"],
            "dim": r["dim"],
            "generator": r.get("generator"),
            "n_chars": r.get("n_chars"),
            **v,
            "text_preview": str(r["text"])[:160].replace("\n", " "),
            "verifiable": "",
            "source_url_or_doi": "",
            "verdict": "",
            "reviewer": "",
            "note": "",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["verification_status", "domain", "query_id", "dim"])
    return out


def summarize_verification(ver: pd.DataFrame) -> pd.DataFrame:
    return (
        ver.groupby(["dim", "verification_status", "recommended_verdict"])
        .size()
        .reset_index(name="count")
        .sort_values(["dim", "count"], ascending=[True, False])
    )
