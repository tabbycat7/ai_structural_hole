"""Frozen real-query pool imported from public QA datasets (DuReader).

Each pool record pairs one *real user question* with the *real web passages*
that were retrieved for it (DuReader ships both), so a single import solves two
validity gaps at once: the query distribution Q comes from real users instead
of 15 hand-written topics, and the competition environment R can use real
passages as distractors while targets stay controlled.

Records are frozen to `data/query_pool/<query_id>.json` (commit them!), same
discipline as `data/base_articles/`: import once, spot-check by hand, never
touch during the experiment.

Supported input formats (auto-detected per line / per record):
  - DuReader 2.0 json-lines: {"question": ..., "documents": [{"paragraphs":
    [...]}, ...]}  (search / zhidao, raw or preprocessed)
  - generic: {"question": ..., "passages": ["...", ...]}
"""
from __future__ import annotations

import json
import random
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..config import DOMAINS, PATHS
from .manipulation_check import detect_profile
from .schema import Article, Query, stable_id

QUESTION_CHARS = (8, 40)
PASSAGE_CHARS = (120, 500)
MIN_PASSAGES = 4
MAX_PASSAGES = 8
# how many candidates per domain to gather before sampling (bounds memory on
# multi-hundred-MB dumps; deterministic given file order)
CANDIDATE_CAP_FACTOR = 20


def pool_dir(root: Optional[Path] = None) -> Path:
    d = (root or PATHS.data_dir) / "query_pool"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Domain classification (keyword-based, favouring precision over recall)
# --------------------------------------------------------------------------- #
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "consumer_product": [
        "耳机", "手机", "电脑", "笔记本", "相机", "冰箱", "洗衣机", "扫地机器人",
        "吸尘器", "电视", "键盘", "鼠标", "显示器", "平板", "音箱", "净水器",
        "空调", "选购", "哪款", "什么牌子", "哪个牌子", "性价比", "值得买",
    ],
    "health": [
        "失眠", "维生素", "血压", "血糖", "感冒", "减肥", "健身", "颈椎", "腰椎",
        "护腰", "睡眠", "营养", "锻炼", "症状", "疫苗", "体检", "心率", "过敏",
        "皮肤", "肠胃", "养生", "康复", "吃什么药",
    ],
    "finance": [
        "基金", "股票", "理财", "定投", "存款", "利率", "信用卡", "贷款", "保险",
        "房贷", "公积金", "社保", "个税", "投资", "收益", "分期", "汇率", "债券",
        "养老金",
    ],
    "academic_qa": [
        "算法", "模型", "论文", "神经网络", "机器学习", "深度学习", "因果推断",
        "统计学", "定理", "原理", "推导", "数学", "物理", "化学", "编程", "代码",
        "Transformer", "数据结构",
    ],
    "travel": [
        "旅游", "旅行", "行程", "景点", "攻略", "签证", "机票", "航班", "酒店",
        "高原", "自驾", "徒步", "露营", "民宿", "火车票", "出行", "自由行",
    ],
}


def classify_domain(question: str) -> Optional[str]:
    """Assign a question to one of the 5 experiment domains (None = discard).

    Counts keyword hits per domain; requires a unique argmax so ambiguous
    questions are dropped rather than misfiled.
    """
    hits = {
        dom: sum(1 for kw in kws if kw in question)
        for dom, kws in DOMAIN_KEYWORDS.items()
    }
    best = max(hits.values())
    if best == 0:
        return None
    winners = [dom for dom, h in hits.items() if h == best]
    return winners[0] if len(winners) == 1 else None


# --------------------------------------------------------------------------- #
# Passage cleaning
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_passage(text: str) -> str:
    text = _TAG_RE.sub("", text or "")
    text = _WS_RE.sub(" ", text).strip()
    return text


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Cut overlong passages at the last sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    cut = max(head.rfind("。"), head.rfind("！"), head.rfind("？"))
    return head[: cut + 1] if cut > 0 else head


def usable_passages(raw_passages: Iterable[str]) -> List[str]:
    """Clean, length-band, truncate and dedup passages (order-preserving)."""
    lo, hi = PASSAGE_CHARS
    out: List[str] = []
    seen: set = set()
    for raw in raw_passages:
        text = clean_passage(raw)
        text = _truncate_at_sentence(text, hi)
        if len(text) < lo:
            continue
        key = _WS_RE.sub("", text)[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= MAX_PASSAGES:
            break
    return out


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
def parse_record(obj: dict) -> Optional[dict]:
    """Extract {"question", "passages"} from one dataset record (or None)."""
    question = (obj.get("question") or "").strip()
    if not question:
        return None
    raw: List[str] = []
    if "documents" in obj:  # DuReader 2.0
        for doc in obj.get("documents") or []:
            raw.extend(doc.get("paragraphs") or [])
    elif "passages" in obj:  # generic
        for p in obj.get("passages") or []:
            raw.append(p if isinstance(p, str) else str(p.get("text", "")))
    if not raw:
        return None
    return {"question": question, "passages": raw}


def iter_dataset_records(path: Path) -> Iterable[dict]:
    """Yield parsed records from a json-lines (or single-JSON-array) file."""
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":  # whole-file JSON array
            for obj in json.load(fh):
                rec = parse_record(obj)
                if rec:
                    yield rec
            return
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rec = parse_record(obj)
            if rec:
                yield rec


# --------------------------------------------------------------------------- #
# Import pipeline
# --------------------------------------------------------------------------- #
def import_queries(
    path: Path,
    per_domain: int = 50,
    domains: Optional[Sequence[str]] = None,
    seed: int = 0,
    root: Optional[Path] = None,
    source: str = "dureader",
) -> Dict[str, Dict[str, int]]:
    """Parse a dataset file into frozen pool records; returns per-domain stats.

    Deterministic given the file, quota and seed. Existing records for the same
    question are overwritten (records are keyed by a stable question hash).
    """
    domains = list(domains or DOMAINS)
    q_lo, q_hi = QUESTION_CHARS
    cap = per_domain * CANDIDATE_CAP_FACTOR

    candidates: Dict[str, List[dict]] = {d: [] for d in domains}
    seen_questions: set = set()
    for rec in iter_dataset_records(Path(path)):
        if all(len(candidates[d]) >= cap for d in domains):
            break
        question = rec["question"]
        if not (q_lo <= len(question) <= q_hi):
            continue
        if question in seen_questions:
            continue
        dom = classify_domain(question)
        if dom not in candidates or len(candidates[dom]) >= cap:
            continue
        passages = usable_passages(rec["passages"])
        if len(passages) < MIN_PASSAGES:
            continue
        seen_questions.add(question)
        candidates[dom].append({"question": question, "passages": passages})

    rng = random.Random(seed)
    stats: Dict[str, Dict[str, int]] = {}
    for dom in domains:
        pool = candidates[dom]
        chosen = rng.sample(pool, min(per_domain, len(pool)))
        for item in chosen:
            record = build_pool_record(item["question"], dom, item["passages"], source=source)
            save_pool_record(record, root=root)
        stats[dom] = {"candidates": len(pool), "imported": len(chosen)}
    return stats


def build_pool_record(
    question: str, domain: str, passages: List[str], source: str = "dureader"
) -> dict:
    qid = stable_id("qpool", domain, question)
    return {
        "id": qid,
        "domain": domain,
        "question": question,
        "factual_core": question,
        "source": source,
        "passages": [
            {
                "text": p,
                "n_chars": len(p),
                # competitor feature strength, usable as an analysis covariate
                "detected_profile": detect_profile(p),
            }
            for p in passages
        ],
    }


def save_pool_record(record: dict, root: Optional[Path] = None) -> Path:
    path = pool_dir(root) / f"{record['id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Loading (frozen pool -> experiment objects)
# --------------------------------------------------------------------------- #
def load_pool_records(root: Optional[Path] = None) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for path in sorted(pool_dir(root).glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("id") and rec.get("question"):
            out[rec["id"]] = rec
    return out


def load_pool_queries(
    per_domain: Optional[int] = None,
    domains: Optional[Sequence[str]] = None,
    root: Optional[Path] = None,
) -> List[Query]:
    """Frozen pool records as `Query` objects (deterministic order)."""
    domains = list(domains or DOMAINS)
    by_domain: Dict[str, List[dict]] = {d: [] for d in domains}
    for rec in load_pool_records(root).values():
        if rec["domain"] in by_domain:
            by_domain[rec["domain"]].append(rec)

    queries: List[Query] = []
    for dom in domains:
        recs = sorted(by_domain[dom], key=lambda r: r["id"])
        if per_domain is not None:
            recs = recs[:per_domain]
        for rec in recs:
            queries.append(
                Query(
                    id=rec["id"],
                    domain=rec["domain"],
                    text=rec["question"],
                    factual_core=rec.get("factual_core") or rec["question"],
                )
            )
    return queries


def load_pool_passages(root: Optional[Path] = None) -> Dict[str, List[dict]]:
    """query_id -> passage dicts ({"text", "n_chars", "detected_profile"})."""
    return {qid: rec.get("passages", []) for qid, rec in load_pool_records(root).items()}


def passage_to_article(query: Query, passage: dict, idx: int) -> Article:
    """Wrap one frozen real passage as a distractor Article."""
    text = passage["text"]
    art = Article(
        id=stable_id("art", query.id, "real_passage", idx, text[:60]),
        query_id=query.id,
        text=text,
        is_target=False,
        meta={"role": "distractor", "generator": "real_passage"},
    )
    art.verified_profile = passage.get("detected_profile") or detect_profile(text)
    return art
