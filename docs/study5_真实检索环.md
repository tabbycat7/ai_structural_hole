# Study 5：真实 RAG 检索环（方案 A）

前四个实验是"把 3 篇候选直接塞进 prompt、强制单选一篇"的受控仿真。Study 5 把它升级为**真实两阶段 RAG**：受控目标文先在一个真实语料库里参与检索，再在自然作答时被（或不被）引用。它衡量的不再只是"被选中"，而是把"结构洞"分解为两个可分别度量的通道。

## 两个通道

- **检索通道** `P_retrieve = P(目标进 top-k)`：某个内容/结构特征是否让文章更容易被真实检索器召回。
- **生成通道** `P_cite|ctx = P(目标被引用 | 目标已在上下文里)`：把目标**强制放进** top-k 上下文，隔离检索环节后，模型自然作答时是否更倾向引用它。
- **端到端** `P_e2e ≈ P_retrieve × P_cite|ctx`：真实产品里"既被检索到又被引用"的综合概率。

强制插入的意义：如果只看自然检索到的情况，目标没被召回时"是否被引用"就无定义、且处理/对照样本不平衡。强制插入让引用结局始终有定义，从而把生成环节的效应干净地从检索环节里剥离出来。

## 因果设计（沿用 Study 1）

- 目标文仍是 `ofat_pairs()` 的成对 control/treatment（一次只改一个 S/O 特征）。
- control 与 treatment 面对**同一个冻结语料库**（等价于 Study 1 的"共享竞争者"原则），配对只差目标特征本身。
- 配对 ATE 按 `query_id` 聚类自助；OFAT 的 EI 用 `scope_col="target_dim"` 限定在各特征自己的配对内（与 Study 1 完全一致）。

## 复用 Study 1 的 LLM 数据（零重复生成）

目标文直接加载 Study 1 已冻结的 LLM 变体，不重新调用大模型：

- 目标文 id 与生成路线无关，用 `build_targets(route="template")` 造出正确 id 的壳子，再从 `data/variant_articles/` 按 `article_id` 取记录、`apply_record` 覆盖冻结正文。
- 无需 gen_client、不联网、不花钱；基线文章经 `data/base_articles/` 自动复用。
- 某目标在库里没有 `generator=="llm"` 记录时，保留模板壳子并在 `targets_manifest.csv` 标记 `template_fallback` 且告警，不静默重生成。正常配置下应 100% `reused`。

竞争者来自真实语料库，所以 Study 5 **不生成任何干扰文**——唯一的 API 开销是第二阶段"读 top-k → 作答并标注引用"的选择实验本身（且走现有缓存）。

## 检索器

`retrieval/retriever.py` 的 `HybridRetriever`：

- **BM25**（`jieba` 分词 + `rank_bm25`）：语料词频/IDF 预计算一次，注入的目标文用同一 IDF 打分，避免逐目标重建索引。
- **向量**（`sentence-transformers`，默认 `BAAI/bge-small-zh-v1.5`）：语料向量预计算并按语料指纹缓存到 `.cache/rag_embeddings/`；query 与目标文实时编码。
- **融合**：`score = alpha·向量cos + (1-alpha)·BM25`（min-max 归一后加权，`alpha` 默认 0.5）。
- 每个 query 的语料得分只算一次，跨该 query 的 16 个目标复用。

## 运行步骤

```bash
# 1) 从冻结题库的真实段落聚合去重，按领域冻结检索语料库（可选 --embed 预热向量）
python -m ai_structural_holes.cli build-corpus

# 2) 跑 Study 5：复用 Study1 冻结目标文，在真实语料上做检索+引用
python -m ai_structural_holes.cli study-rag \
  --models deepseek/deepseek-v4-flash \
  --per-domain 20 --top-k 8 --concurrency 50 --retriever hybrid --alpha 0.5

# 离线冒烟（无需密钥/向量模型）：
python -m ai_structural_holes.cli study-rag --mock --retriever bm25 --per-domain 1 --top-k 5
```

## 产出

- 共享（与模型无关）：`outputs/study_rag/targets_manifest.csv`（目标文复用清单）、`retrieval.csv`（每个目标的 `retrieved`/`target_rank`/`target_score`）、`ate_retrieved.csv`（检索通道 ATE）。
- 每个模型子目录：`trials.csv`（引用 trial）、`ate_cite.csv`（生成通道 ATE）、`ei_leverage.csv`、`e2e.csv`（三通道概率与效应的分解表）。

## 效度要点

- control/treatment 共享同一冻结语料 → 配对只差目标特征。
- 强制插入把检索环节与生成环节解耦，引用结局始终有定义。
- 语料与向量全部冻结（`data/rag_corpus/` 建议提交进版本管理），检索确定、可复现；仅生成阶段调 API 并复用缓存。
- 沿用 `parse_ok` 缺失处理（引用阶段 `cited` 为空是"引用了但没用目标"的合法结局，不计为解析失败）与按 `query_id` 聚类。
- BM25 对纯排版类特征（O1/O2/O3）几乎不敏感属预期——这类特征的结构洞主要体现在生成通道，正是两通道分解要揭示的现象。
- 检索器与向量模型版本、`top_k`、`alpha` 都是外部效度边界，报告时需说明。
