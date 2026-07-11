# 实验数据（本地生成，不入库）

本目录下的冻结数据体积较大或由 LLM 生成，默认不提交到 Git。按以下步骤在本地重建：

## 1. 导入 DuReader 题库

从 [DuReader 2.0](https://github.com/baidu/DuReader) 下载 `search.dev.json`（约 140MB），放到项目根目录后：

```bash
python -m ai_structural_holes.cli import-queries --file search.dev.json --per-domain 100
```

生成 `data/query_pool/`（500 题，五领域各 100 题）。

## 2. 生成基线与变体文章

```bash
python -m ai_structural_holes.cli gen-base --model deepseek/deepseek-chat --query-source pool --per-domain 100
python -m ai_structural_holes.cli gen-variants --model deepseek/deepseek-chat --query-source pool --per-domain 100
```

## 3. Study 5/6 检索语料（可选）

```bash
python -m ai_structural_holes.cli build-corpus
```

生成 `data/rag_corpus/`。
