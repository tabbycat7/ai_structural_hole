"""Codebook for the semantic (S) and structural (O) feature dimensions.

Each dimension is operationalized as an ordinal/binary factor with explicit
levels, a definition, the raw elements it absorbs, positive/negative examples,
and a *manipulation-check* rule used to verify that (a) the target dimension was
actually changed and (b) only the target dimension changed.

These objects are the single source of truth used by data generation, the study
designs, manipulation checking, and the causal/EI analysis (factor coding).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Layer(str, Enum):
    SEMANTIC = "S"  # what the content says
    STRUCTURAL = "O"  # how the content is organized


@dataclass(frozen=True)
class Level:
    """One ordinal level of a dimension.

    code: integer used in factor coding (0 = baseline/absent, higher = stronger).
    label: short machine label.
    description: human description of what this level looks like in text.
    """

    code: int
    label: str
    description: str


@dataclass(frozen=True)
class Dimension:
    id: str  # e.g. "S1"
    layer: Layer
    name: str  # human name
    definition: str
    absorbs: List[str]  # raw elements absorbed into this dimension
    levels: List[Level]
    manipulation_check: str  # rule for verifying the manipulation
    positive_example: str
    negative_example: str
    # Markers that an automated manipulation-check classifier can look for.
    presence_markers: List[str] = field(default_factory=list)

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def is_binary(self) -> bool:
        return self.n_levels == 2

    def codes(self) -> List[int]:
        return [lv.code for lv in self.levels]

    def level_by_code(self, code: int) -> Level:
        for lv in self.levels:
            if lv.code == code:
                return lv
        raise KeyError(f"{self.id} has no level with code {code}")

    def baseline_code(self) -> int:
        return min(self.codes())

    def top_code(self) -> int:
        return max(self.codes())


def _bin(absent: str, present: str) -> List[Level]:
    return [
        Level(0, "absent", absent),
        Level(1, "present", present),
    ]


# --------------------------------------------------------------------------- #
# Semantic layer S1..S4
# --------------------------------------------------------------------------- #
S1 = Dimension(
    id="S1",
    layer=Layer.SEMANTIC,
    name="证据坚实度 (evidence solidity)",
    definition="文章为核心主张提供事实依据、来源背书或经验性支持的程度。",
    absorbs=["具体统计数据", "第三方来源", "权威机构", "用户评价"],
    levels=[
        Level(0, "none", "无任何依据，仅作主张。"),
        Level(1, "vague", "笼统断言，如'研究表明''很多人认为'，无可核验来源。"),
        Level(2, "solid", "具体统计数据 + 可指认的第三方/权威来源 + 用户评价。"),
    ],
    manipulation_check=(
        "检查是否出现可核验的数字、命名来源或机构;仅改变证据强度而不改变主题、"
        "结论方向与文本长度(±10%)。"
    ),
    positive_example="据 2023 年 JAMA 一项 N=1,204 的随机对照试验，组 A 有效率为 78%(95%CI 74-82)。",
    negative_example="这个方法效果很好，大家都说不错。",
    presence_markers=["%", "CI", "N=", "研究", "数据", "来源", "机构", "评价", "et al"],
)

S2 = Dimension(
    id="S2",
    layer=Layer.SEMANTIC,
    name="视角辩证度 (dialectical balance)",
    definition="文章是否呈现风险、局限、反方观点、不确定性和适用边界。",
    absorbs=["风险提示", "反方观点", "局限", "适用边界", "不确定性"],
    levels=_bin(
        absent="仅单边主张，不提风险或局限。",
        present="明确给出风险/局限/反方观点/适用边界/不确定性。",
    ),
    manipulation_check=(
        "检查是否出现'但是/然而/局限/风险/不适用于'等辩证内容;仅增删辩证段落，"
        "不改变主结论与证据。"
    ),
    positive_example="该方案在高并发下表现优异，但在小数据量场景反而带来额外开销，且不适用于离线批处理。",
    negative_example="该方案在任何场景下都是最佳选择。",
    presence_markers=["但是", "然而", "局限", "风险", "不适用", "反方", "不确定", "权衡"],
)

S3 = Dimension(
    id="S3",
    layer=Layer.SEMANTIC,
    name="领域专业性 (domain expertise)",
    definition="文章是否使用领域相关的专业术语、概念框架和机制解释。",
    absorbs=["专业术语", "概念框架", "机制解释"],
    levels=_bin(
        absent="通俗表述，无专业术语或机制解释。",
        present="使用领域术语 + 概念框架 + 机制层面的解释。",
    ),
    manipulation_check=(
        "检查是否出现领域术语与机制性解释('因为…机制…导致…');仅改变专业化表达，"
        "保持事实内核与结论不变。"
    ),
    positive_example="其加速源于 KV-cache 复用降低了自回归解码的显存带宽瓶颈，从而提升 tokens/s。",
    negative_example="它跑得更快，因为优化得好。",
    presence_markers=["机制", "因为", "framework", "范式", "术语"],
)

S4 = Dimension(
    id="S4",
    layer=Layer.SEMANTIC,
    name="主张明确性 (claim clarity)",
    definition="文章是否直接回应任务，并提供明确结论、核心优势和选择理由。",
    absorbs=["明确结论", "品牌/产品/方案核心优势", "选择理由"],
    levels=_bin(
        absent="结论含糊，不直接回应任务，无明确推荐理由。",
        present="直接给出明确结论 + 核心优势 + 选择理由。",
    ),
    manipulation_check=(
        "检查是否存在一句可定位的明确结论与推荐理由;仅改变结论明确性，不改变证据与立场。"
    ),
    positive_example="结论:针对你的预算优先选 B 方案，因为它在成本和可维护性上同时占优。",
    negative_example="这取决于很多因素，不同情况各有优劣，很难一概而论。",
    presence_markers=["结论", "推荐", "建议选择", "首选", "理由"],
)

# --------------------------------------------------------------------------- #
# Structural layer O1..O4
# --------------------------------------------------------------------------- #
O1 = Dimension(
    id="O1",
    layer=Layer.STRUCTURAL,
    name="信息呈现形态 (presentation form)",
    definition="文本将信息组织为连续段落或离散单元的程度。",
    absorbs=["段落", "列表", "编号", "表格"],
    levels=_bin(
        absent="连续散文段落。",
        present="列表/编号/表格等离散单元。",
    ),
    manipulation_check=(
        "检查是否使用列表/编号/表格标记;仅改变排版形态，词句信息内容保持等价。"
    ),
    positive_example="- 优点1\n- 优点2\n- 优点3",
    negative_example="它有三个优点，分别体现在性能、成本与易用性等方面，彼此关联。",
    presence_markers=["- ", "* ", "1.", "2.", "|", "\n- "],
)

O2 = Dimension(
    id="O2",
    layer=Layer.STRUCTURAL,
    name="宏观信息顺序 (macro order)",
    definition="摘要、结论、证据和关键信息在篇章中的先后位置。",
    absorbs=["摘要先行", "结论前置", "证据前置", "结论后置"],
    levels=_bin(
        absent="结论后置(铺垫在前，结论在末尾)。",
        present="结论/证据前置(摘要先行，先给结论再展开)。",
    ),
    manipulation_check=(
        "检查首段是否已给出结论/摘要;仅重排信息顺序，不增删信息单元。"
    ),
    positive_example="结论先行:推荐 B。理由如下……(随后展开证据)",
    negative_example="(三段铺垫之后)……综上所述，最终我们才得出推荐 B。",
    presence_markers=["结论先行", "摘要", "TL;DR", "综上(末尾)"],
)

O3 = Dimension(
    id="O3",
    layer=Layer.STRUCTURAL,
    name="逻辑结构显性化 (logical explicitness)",
    definition="文本是否显式标出不同信息单元的论证功能。",
    absorbs=["小标题", "适用场景", "优点", "局限", "结论", "问题—分析—结论"],
    levels=_bin(
        absent="无功能标注的连续文本。",
        present="使用小标题/功能标签(优点/局限/结论/适用场景)显式标注论证功能。",
    ),
    manipulation_check=(
        "检查是否存在功能性小标题或标签;仅添加结构标签，不改变标签下的实质内容。"
    ),
    positive_example="## 优点\n……\n## 局限\n……\n## 结论\n……",
    negative_example="(全篇无标题，所有内容混在连续段落中)",
    presence_markers=["#", "##", "【", "优点:", "局限:", "结论:", "适用场景"],
)

O4 = Dimension(
    id="O4",
    layer=Layer.STRUCTURAL,
    name="证据—主张邻近性 (evidence-claim proximity)",
    definition="证据与其支持的主张在文本中的空间距离和语义绑定程度。",
    absorbs=["远距离证据", "邻近证据", "绑定证据"],
    levels=[
        Level(0, "distant", "证据与所支持的主张相隔很远(不同段落/章节)。"),
        Level(1, "adjacent", "证据紧邻主张(同段相邻句)。"),
        Level(2, "bound", "证据与主张显式绑定(同句/括注内,如'X(证据:…)')。"),
    ],
    manipulation_check=(
        "检查每条主张与其支撑证据的距离/绑定;仅移动证据位置与绑定方式，不改变证据或主张本身。"
    ),
    positive_example="B 方案更省成本(证据:实测较 A 低 31%，来源:内部基准 2024Q1)。",
    negative_example="B 方案更省成本。……(三段之后)……另外我们曾测得某方案成本较低。",
    presence_markers=["(证据:", "（证据:", "即(", "如下数据"],
)


SEMANTIC_DIMENSIONS: List[Dimension] = [S1, S2, S3, S4]
STRUCTURAL_DIMENSIONS: List[Dimension] = [O1, O2, O3, O4]
ALL_DIMENSIONS: List[Dimension] = SEMANTIC_DIMENSIONS + STRUCTURAL_DIMENSIONS
DIMENSIONS: Dict[str, Dimension] = {d.id: d for d in ALL_DIMENSIONS}


def get_dimension(dim_id: str) -> Dimension:
    return DIMENSIONS[dim_id]


def semantic_ids() -> List[str]:
    return [d.id for d in SEMANTIC_DIMENSIONS]


def structural_ids() -> List[str]:
    return [d.id for d in STRUCTURAL_DIMENSIONS]


def all_ids() -> List[str]:
    return list(DIMENSIONS.keys())


def baseline_profile() -> Dict[str, int]:
    """The canonical 'all-baseline' feature profile (every dimension at code 0)."""
    return {d.id: d.baseline_code() for d in ALL_DIMENSIONS}


def top_profile() -> Dict[str, int]:
    """The 'all-strong' feature profile (every dimension at top code)."""
    return {d.id: d.top_code() for d in ALL_DIMENSIONS}


if __name__ == "__main__":  # quick manual inspection
    for d in ALL_DIMENSIONS:
        levels = ", ".join(f"{lv.code}={lv.label}" for lv in d.levels)
        print(f"[{d.id}] {d.name} | levels: {levels}")
