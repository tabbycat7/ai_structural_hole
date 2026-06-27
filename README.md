# AI 结构洞 (AI Structural Holes)

RAG 多候选竞争场景下的因果实验框架:研究文章的**语义特征 S** 与**结构特征 O** 如何
**因果性地**提高大模型在多篇候选中"选中/引用"某篇的概率。方法学沿用「EQ 归因测评框架」
同款路线 —— DAG 建模 → 识别混杂集 → do 演算后门调整(定理 1) → 用 **EI(有效信息
Effective Information)** 量化各因素的因果影响权重。所有模型调用统一经 **OpenRouter**。

> 实现细节与公式见 `docs/`、计划文件以及各模块 docstring。

## 安装

```bash
pip install -r requirements.txt
# 或可编辑安装(含 CLI 入口)
pip install -e .
```

配置 API key(可选;不配置则自动回退到离线 MockClient):

```bash
cp .env.example .env   # 填入 OPENROUTER_API_KEY
```

## 快速开始(离线 Mock,无需 key)

```bash
python -m ai_structural_holes.cli graph          # 打印 DAG 与混杂集
python -m ai_structural_holes.cli power           # 样本量/功效分析
python -m ai_structural_holes.cli study1 --mock   # 单特征配对干预 + 每维 ATE/EI
python -m ai_structural_holes.cli study2 --mock --n-points 16
python -m ai_structural_holes.cli study3 --mock
python -m ai_structural_holes.cli study4 --mock
```

接入真实模型(经 OpenRouter):

```bash
python -m ai_structural_holes.cli study1 --models openai/gpt-4o,deepseek/deepseek-chat --seeds 3
```

输出(CSV + 图)写入 `outputs/<study>/`。

## 因果路线(计划第 2、7 节)

DAG 边: `Q→S, Q→O, S→O, Q→Y, S→Y, O→Y, M→Y, R→Y`。混杂集:
`A_S={Q}`、`A_O={Q,S}`、`A_M=A_R={}`。

1. 去偏得 `P(Y|do(X))`:实验 do 路线(随机化 → `P(Y|X)`)或 SCM 后门调整(定理 1)
   —— `ai_structural_holes/causal/backdoor.py`。
2. `EI(X→Y) = (1/|X|) Σ_x KL(P(Y|do(X=x)) || P̄(Y))`,分解为 determinism − degeneracy,
   归一化后得跨因素"结构洞杠杆排序" —— `ai_structural_holes/causal/ei.py`。

## 四个 Study(计划第 4 节)

- Study 1 单特征配对干预(OFAT)→ 每维 ATE 与 EI。
- Study 2 分数析因 → 主效应 + 关键 S×O 交互(S1×O4, S1×O2, O1×O3)。
- Study 3 跨 M/领域/提示/R 泛化 → 分层 EI + 跨模型一致性(Kendall W/Spearman)。
- Study 4 反向对抗 → 真 vs 伪特征,欺骗增益、脆弱性、ΔEI。

## 目录结构

```
ai_structural_holes/
  codebook.py            # S1-S4 / O1-O4 维度、档位、操纵检查规则
  config.py              # 路径、模型清单、领域、提示风格
  poweranalysis.py       # 功效/样本量
  cli.py                 # 命令行入口
  llm/                   # OpenRouter 统一调用层 + 缓存 + Mock
  data/                  # 数据 schema、变体生成、操纵检查
  task/                  # RAG 选择任务:提示模板、候选集、位置平衡、解析
  experiment/            # 试验编排 runner -> tidy DataFrame
  causal/                # DAG/混杂集、后门调整、EI
  analysis/              # ATE、回归、异质/调节/中介、位置偏差、指标、画图
  studies/              # 四个 study 的设计与编排
docs/validity_threats.md # 效度威胁→对策→代码 映射
tests/                   # EI/后门/操纵检查/功效/study 冒烟测试
```

## 测试

```bash
python -m pytest -q
```

## 重要说明

- **MockClient 不是真实模型**:它用"位置 + 内容标记"的透明打分模拟选择,仅用于离线打通
  全链路(解析→ATE→后门→EI)。真实结论必须经 OpenRouter 调真实模型获得。
- 因 `S→O` 依赖,O 维度优先走 SCM 后门路线;OFAT 中 O4 只在 S1 存在时操纵。中介/交互分析
  (`mediation_proportion`)应在 Study 2 的析因数据上运行(S1、O4 联合变化)。
- 不在代码中硬编码密钥;`call_model` 内置重试/限速退避与按请求哈希的磁盘缓存(去重省钱、可复现)。
