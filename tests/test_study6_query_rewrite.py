"""Study 6 (query-rewrite-driven RAG): parsing, multi-query fusion, e2e smoke."""
import glob
from types import SimpleNamespace

import pytest

from ai_structural_holes.studies.study6_query_rewrite import _natural_topk_ids
from ai_structural_holes.task.protocol import parse_rewrite


# --------------------------------------------------------------------------- #
# _natural_topk_ids: real end-to-end top-k (no forced insert, no counterbalance)
# --------------------------------------------------------------------------- #
def _res(top_k_doc_ids, target_rank):
    return SimpleNamespace(top_k_doc_ids=list(top_k_doc_ids), target_rank=target_rank)


def test_natural_topk_target_present_at_true_rank():
    # target retrieved at rank 2 of a k=5 window -> sits at index 2, one corpus
    # doc drops off the tail so the window stays size k.
    res = _res(["c0", "c1", "c2", "c3", "c4"], target_rank=2)
    ids = _natural_topk_ids(res, "TGT", k=5)
    assert ids == ["c0", "c1", "TGT", "c2", "c3"]
    assert len(ids) == 5
    assert ids.index("TGT") == 2


def test_natural_topk_target_first_rank():
    res = _res(["c0", "c1", "c2", "c3", "c4"], target_rank=0)
    ids = _natural_topk_ids(res, "TGT", k=5)
    assert ids == ["TGT", "c0", "c1", "c2", "c3"]


def test_natural_topk_target_absent_when_not_retrieved():
    # target_rank >= k -> target absent, candidate set is pure corpus top-k.
    res = _res(["c0", "c1", "c2", "c3", "c4"], target_rank=7)
    ids = _natural_topk_ids(res, "TGT", k=5)
    assert ids == ["c0", "c1", "c2", "c3", "c4"]
    assert "TGT" not in ids


# --------------------------------------------------------------------------- #
# parse_rewrite
# --------------------------------------------------------------------------- #
def test_parse_rewrite_plain_json():
    out = parse_rewrite('{"queries": ["高血压 饮食", "血压 控制"]}', "原始问题")
    assert out["parse_ok"] is True
    assert out["queries"] == ["高血压 饮食", "血压 控制"]


def test_parse_rewrite_code_fence_and_dedup():
    raw = '```json\n{"queries": ["a", "a", "b"]}\n```'
    out = parse_rewrite(raw, "原始问题")
    assert out["parse_ok"] is True
    assert out["queries"] == ["a", "b"]  # deduped, order preserved


def test_parse_rewrite_empty_list_falls_back():
    out = parse_rewrite('{"queries": []}', "原始问题")
    assert out["parse_ok"] is False
    assert out["queries"] == ["原始问题"]


def test_parse_rewrite_garbage_falls_back():
    out = parse_rewrite("模型胡言乱语，没有 JSON", "原始问题")
    assert out["parse_ok"] is False
    assert out["queries"] == ["原始问题"]


def test_parse_rewrite_respects_max_queries():
    out = parse_rewrite('{"queries": ["a","b","c","d"]}', "orig", max_queries=2)
    assert out["queries"] == ["a", "b"]


# --------------------------------------------------------------------------- #
# retrieve_multi fusion
# --------------------------------------------------------------------------- #
def _bm25_retriever():
    pytest.importorskip("jieba")
    pytest.importorskip("rank_bm25")
    from ai_structural_holes.retrieval.corpus import CorpusDoc
    from ai_structural_holes.retrieval.retriever import HybridRetriever

    docs = [
        CorpusDoc(doc_id="d1", domain="health", text="高血压 患者 应该 低盐 饮食 控制 血压", src_query_id="q"),
        CorpusDoc(doc_id="d2", domain="health", text="糖尿病 患者 需要 监测 血糖 水平", src_query_id="q"),
        CorpusDoc(doc_id="d3", domain="health", text="旅行 攻略 与 机票 预订 技巧", src_query_id="q"),
    ]
    return HybridRetriever(docs, mode="bm25")


def test_retrieve_multi_single_query_matches_retrieve():
    retr = _bm25_retriever()
    target = "高血压 低盐 饮食 建议"
    single = retr.retrieve("高血压 饮食", target, k=2)
    multi = retr.retrieve_multi(["高血压 饮食"], target, k=2)
    assert multi.target_rank == single.target_rank
    assert multi.retrieved == single.retrieved
    assert multi.competitor_doc_ids == single.competitor_doc_ids


def test_retrieve_multi_empty_falls_back_to_target_text():
    retr = _bm25_retriever()
    # empty query list must not crash; falls back to using target text as query
    res = retr.retrieve_multi([], "高血压 低盐 饮食", k=2)
    assert res.target_rank >= 0


def test_retrieve_multi_max_vs_mean_are_valid():
    retr = _bm25_retriever()
    target = "高血压 低盐 饮食 建议"
    queries = ["高血压 饮食", "旅行 机票"]
    res_max = retr.retrieve_multi(queries, target, k=2, fuse="max")
    res_mean = retr.retrieve_multi(queries, target, k=2, fuse="mean")
    assert 0 <= res_max.target_rank <= 3
    assert 0 <= res_mean.target_rank <= 3


def test_retrieve_multi_rejects_unknown_fuse():
    retr = _bm25_retriever()
    with pytest.raises(ValueError):
        retr.retrieve_multi(["q"], "t", k=2, fuse="sum")


# --------------------------------------------------------------------------- #
# run_study6 end-to-end smoke (mock client, bm25, offline)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not glob.glob("data/rag_corpus/*.jsonl") or not glob.glob("data/query_pool/*.json"),
    reason="frozen corpus / query pool not built",
)
def test_run_study6_smoke_mock():
    pytest.importorskip("jieba")
    pytest.importorskip("rank_bm25")
    from ai_structural_holes.studies import retrieval_by_model, run_study6

    res = run_study6(
        models=["mock/model-a"],
        per_domain=1,
        domains=["health"],
        seeds=(0,),
        top_k=5,
        retriever="bm25",
        n_queries=2,
        fuse="max",
        query_source="pool",
        mock=True,
        progress=False,
        concurrency=1,
    )
    # rewrite audit exists and carries the model column
    assert not res.rewrite_frame.empty
    assert set(["model", "query_id", "rewritten_queries"]).issubset(res.rewrite_frame.columns)
    # retrieval is model-dependent -> frame carries a model column
    assert "model" in res.retrieval_frame.columns
    assert (res.retrieval_frame["model"] == "mock/model-a").all()
    # cross-model retrieval summary computes without error
    rbm = retrieval_by_model(res.retrieval_frame)
    assert "model" in rbm.columns or rbm.empty

    # No position counterbalancing: each target yields at most ONE generation
    # trial (per seed/prompt), so gen rows must not exceed retrieval rows. The
    # old forced-insert design would have produced up to top_k x more.
    assert len(res.gen_frame) <= len(res.retrieval_frame)

    # Real end-to-end: a target that was NOT retrieved (retrieved==0) is absent
    # from its candidate set, so it can never be cited (y==0).
    keys = ["query_id", "model", "pair_id", "role", "target_dim"]
    if (
        not res.gen_frame.empty
        and "retrieved" in res.retrieval_frame.columns
        and set(keys).issubset(res.gen_frame.columns)
        and set(keys).issubset(res.retrieval_frame.columns)
    ):
        not_retrieved = res.retrieval_frame[res.retrieval_frame["retrieved"] == 0]
        if not not_retrieved.empty:
            merged = res.gen_frame.merge(not_retrieved[keys], on=keys, how="inner")
            if not merged.empty:
                assert (merged["y"] == 0).all()


def test_study6_tables_returns_triple():
    """study6_tables now returns (ate_retrieved, ate_e2e, ei) -- no product e2e."""
    import pandas as pd

    from ai_structural_holes.study_output import study6_tables

    out = study6_tables(pd.DataFrame(), pd.DataFrame())
    assert len(out) == 3
    assert all(isinstance(t, pd.DataFrame) for t in out)


@pytest.mark.skipif(
    not glob.glob("data/rag_corpus/*.jsonl") or not glob.glob("data/query_pool/*.json"),
    reason="frozen corpus / query pool not built",
)
def test_study6_resume_completes_without_duplicates(tmp_path):
    """Partial run + --resume补跑: no duplicate trial_id, full coverage, rewrites reused."""
    pytest.importorskip("jieba")
    pytest.importorskip("rank_bm25")
    import pandas as pd

    from ai_structural_holes.studies import run_study6
    from ai_structural_holes.study_output import StudyModelSink, study_model_dir

    models = ["mock/model-a"]
    kwargs = dict(
        per_domain=1, domains=["health"], seeds=(0,), top_k=5,
        retriever="bm25", n_queries=2, fuse="max", query_source="pool",
        mock=True, progress=False, concurrency=1,
    )

    # Full baseline run to know the complete trial_id set.
    sink_full = StudyModelSink("study6", models, tmp_path, refresh_every=9999, refresh_sec=9999.0)
    run_study6(models=models, output_sink=sink_full, **kwargs)
    sink_full.finalize()
    out_dir = study_model_dir("study6", "mock/model-a", tmp_path)
    full = pd.read_csv(out_dir / "trials.csv")
    full_ids = set(full["trial_id"])
    assert len(full) == len(full_ids)  # baseline itself has no dupes
    assert len(full) >= 2

    # Simulate an interrupted run: keep only the first half of trials on disk.
    half = full.iloc[: max(1, len(full) // 2)].copy()
    half.to_csv(out_dir / "trials.csv", index=False, encoding="utf-8-sig")
    rewrites_before = (out_dir / "rewrites.csv").read_text(encoding="utf-8-sig")

    # Resume: reuse rewrites, skip completed, backfill the rest.
    sink_resume = StudyModelSink(
        "study6", models, tmp_path, refresh_every=9999, refresh_sec=9999.0, resume=True
    )
    run_study6(models=models, output_sink=sink_resume, resume=True, **kwargs)
    sink_resume.finalize()

    resumed = pd.read_csv(out_dir / "trials.csv")
    resumed_ids = list(resumed["trial_id"])
    assert len(resumed_ids) == len(set(resumed_ids))  # no duplicates after resume
    assert set(resumed_ids) == full_ids  # full coverage restored
    # rewrites were reused (not re-appended / duplicated)
    assert (out_dir / "rewrites.csv").read_text(encoding="utf-8-sig") == rewrites_before
