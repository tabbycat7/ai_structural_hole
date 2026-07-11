"""Tests for incremental study output and disabled LLM disk cache."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from unittest import mock

import pandas as pd

from ai_structural_holes.data.schema import Article, CandidateSet, Query
from ai_structural_holes.experiment.incremental_output import (
    IncrementalCsvWriter,
    PeriodicAnalysisRefresher,
)
from ai_structural_holes.experiment.runner import ExperimentRunner
from ai_structural_holes.llm.cache import DiskCache, NullDiskCache, resolve_llm_cache
from ai_structural_holes.llm.client import OpenAICompatibleClient
from ai_structural_holes.studies import run_study1, run_study6
from ai_structural_holes.study_output import StudyModelSink, _analysis_frame


def test_incremental_csv_writer_concurrent_append(tmp_path):
    path = tmp_path / "trials.csv"
    writer = IncrementalCsvWriter(path)

    def _append(i: int):
        writer.append_row({"id": i, "value": f"row-{i}"})

    threads = [threading.Thread(target=_append, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    assert set(rows[0].keys()) == {"id", "value"}


def test_incremental_csv_writer_resume_appends_without_truncating(tmp_path):
    path = tmp_path / "trials.csv"
    w1 = IncrementalCsvWriter(path, truncate_on_init=True)
    w1.append_row({"trial_id": "a", "y": 1})
    w1.append_row({"trial_id": "b", "y": 0})

    # Reopen in resume mode: existing rows kept, header not rewritten, appends add.
    w2 = IncrementalCsvWriter(path, truncate_on_init=False)
    w2.append_row({"trial_id": "c", "y": 1})

    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["trial_id"] for r in rows] == ["a", "b", "c"]
    # exactly one header line
    assert path.read_text(encoding="utf-8-sig").count("trial_id") == 1


def test_incremental_csv_writer_resume_drops_unknown_columns(tmp_path):
    # A legacy file whose header lacks a column added later: new keys are dropped
    # safely (no misalignment) rather than shifting columns.
    path = tmp_path / "trials.csv"
    w1 = IncrementalCsvWriter(path, truncate_on_init=True)
    w1.append_row({"trial_id": "a", "y": 1})

    w2 = IncrementalCsvWriter(path, truncate_on_init=False)
    w2.append_row({"trial_id": "b", "y": 0, "api_error": "content_filter"})

    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["trial_id"] for r in rows] == ["a", "b"]
    assert "api_error" not in rows[0]


def _mini_runner_inputs():
    q = Query(id="q1", domain="health", text="低盐饮食建议")
    tgt = Article(id="t1", query_id="q1", text="目标文本", is_target=True,
                  meta={"target_dim": "S1", "role": "treatment", "pair_id": "p1"})
    comp = Article(id="c1", query_id="q1", text="竞争文本")
    articles = {"t1": tgt, "c1": comp}
    queries = {"q1": q}
    cs = CandidateSet(query_id="q1", ordered_ids=["t1", "c1"], target_id="t1",
                      competitor_quality="real")
    return queries, articles, [cs]


class _FilterClient:
    """Mock client that raises a content_filter 400 on every call."""

    def call(self, **kw):
        raise RuntimeError(
            "Error code: 400 - content_filter: request considered high risk"
        )


class _OkClient:
    def __init__(self):
        self.n = 0

    def call(self, **kw):
        self.n += 1
        return type("R", (), {"text": '{"choice":"A"}', "cached": False})()


def test_runner_tolerates_content_filter_and_marks_api_error():
    queries, articles, sets = _mini_runner_inputs()
    runner = ExperimentRunner(client=_FilterClient(), models=["m/x"],
                              output_mode="cite", concurrency=1)
    frame = runner.run(queries, articles, sets, progress=False)
    assert len(frame) == 1  # did not crash
    assert frame.iloc[0]["api_error"] == "content_filter"
    assert frame.iloc[0]["parse_ok"] == 0


def test_runner_skip_predicate_prevents_api_call():
    queries, articles, sets = _mini_runner_inputs()
    client = _OkClient()
    runner = ExperimentRunner(client=client, models=["m/x"],
                              output_mode="cite", concurrency=1)
    frame = runner.run(
        queries, articles, sets, progress=False,
        skip=lambda cs, model, style, seed: True,  # skip everything
    )
    assert frame.empty
    assert client.n == 0  # skipped combo never triggered an API call


def test_runner_skip_predicate_partial():
    queries, articles, sets = _mini_runner_inputs()
    client = _OkClient()
    runner = ExperimentRunner(client=client, models=["m/x"],
                              output_mode="cite", concurrency=1)
    # skip only when target_id == 't1' (our only set) -> nothing runs
    frame = runner.run(
        queries, articles, sets, progress=False,
        skip=lambda cs, model, style, seed: cs.target_id == "nope",  # skip none
    )
    assert len(frame) == 1
    assert client.n == 1


def test_sink_completed_row_keys_and_dedupe(tmp_path):
    from ai_structural_holes.study_output import study_model_dir

    sink = StudyModelSink("study6", ["m/x"], tmp_path)
    out = study_model_dir("study6", "m/x", tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    key_cols = ["query_id", "model", "prompt_style", "seed", "target_dim", "role", "pair_id"]
    # Two distinct trials, with the second duplicated (as a broken resume would).
    rows = [
        {"query_id": "q1", "model": "m/x", "prompt_style": "neutral", "seed": 0,
         "target_dim": "S1", "role": "treatment", "pair_id": "p1", "y": 1},
        {"query_id": "q1", "model": "m/x", "prompt_style": "neutral", "seed": 0,
         "target_dim": "S1", "role": "control", "pair_id": "p1", "y": 0},
        {"query_id": "q1", "model": "m/x", "prompt_style": "neutral", "seed": 0,
         "target_dim": "S1", "role": "control", "pair_id": "p1", "y": 0},  # dup
    ]
    pd.DataFrame(rows).to_csv(out / "trials.csv", index=False, encoding="utf-8-sig")

    removed = sink.dedupe_trials("m/x", key_cols)
    assert removed == 1
    df = pd.read_csv(out / "trials.csv")
    assert len(df) == 2
    assert "api_error" in df.columns  # added for future append alignment

    keys = sink.completed_row_keys("m/x", key_cols)
    assert ("q1", "m/x", "neutral", "0", "S1", "treatment", "p1") in keys
    assert ("q1", "m/x", "neutral", "0", "S1", "control", "p1") in keys
    assert len(keys) == 2


def test_analysis_frame_excludes_api_error_rows():
    df = pd.DataFrame([
        {"y": 1, "parse_ok": 1, "api_error": ""},
        {"y": 0, "parse_ok": 1, "api_error": "content_filter"},
        {"y": 1, "parse_ok": 0, "api_error": ""},
    ])
    out = _analysis_frame(df)
    assert len(out) == 1
    assert out.iloc[0]["y"] == 1


def test_null_disk_cache_client_no_files():
    client = OpenAICompatibleClient(
        base_url="http://example.invalid",
        api_key="test-key",
        cache=NullDiskCache(),
    )
    with mock.patch.object(client, "_raw_call", return_value={"text": "ok", "usage": {}}):
        messages = [{"role": "user", "content": "ping"}]
        r1 = client.call(model="openai/gpt-4o", messages=messages, max_tokens=16)
        r2 = client.call(model="openai/gpt-4o", messages=messages, max_tokens=16)
    assert r1.cached is False
    assert r2.cached is False


def test_resolve_llm_cache_env(monkeypatch):
    monkeypatch.delenv("ASH_LLM_CACHE", raising=False)
    assert isinstance(resolve_llm_cache(), NullDiskCache)
    monkeypatch.setenv("ASH_LLM_CACHE", "1")
    assert isinstance(resolve_llm_cache(), DiskCache)


def test_periodic_analysis_refresher_triggers():
    calls: list[str] = []

    def _cb():
        calls.append("ok")

    refresher = PeriodicAnalysisRefresher(_cb, refresh_every=3, refresh_sec=9999.0)
    refresher.tick()
    refresher.tick()
    assert calls == []
    refresher.tick()
    assert calls == ["ok"]

    calls.clear()
    slow = PeriodicAnalysisRefresher(_cb, refresh_every=9999, refresh_sec=0.05)
    slow.tick()
    time.sleep(0.08)
    slow.tick()
    assert calls == ["ok"]


def test_study1_incremental_sink_smoke(tmp_path):
    models = ["mock/a"]
    sink = StudyModelSink("study1", models, tmp_path, refresh_every=1, refresh_sec=0.0)
    res = run_study1(
        models,
        per_domain=1,
        mock=True,
        progress=False,
        output_sink=sink,
        use_llm_cache=False,
    )
    saved = sink.finalize()
    assert len(res.frame) > 0
    assert saved
    trials = pd.read_csv(saved[0] / "trials.csv")
    assert len(trials) == len(res.frame)
    assert (saved[0] / "ate.csv").exists()


def test_study6_rewrite_incremental_smoke(tmp_path):
    models = ["mock/a"]
    sink = StudyModelSink("study6", models, tmp_path, refresh_every=9999, refresh_sec=9999.0)
    with mock.patch(
        "ai_structural_holes.studies.study6_query_rewrite.load_corpus",
        return_value=[],
    ):
        res = run_study6(
            models,
            per_domain=1,
            mock=True,
            progress=False,
            output_sink=sink,
            use_llm_cache=False,
        )
    assert not res.rewrite_frame.empty
    rewrite_path = tmp_path / "study6" / "mock__a" / "rewrites.csv"
    assert rewrite_path.exists()
    rw = pd.read_csv(rewrite_path)
    assert len(rw) == len(res.rewrite_frame)
