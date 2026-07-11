"""Per-model output paths and persistence for study CLI commands."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import pandas as pd

from .analysis.ate import ate_table
from .analysis.metrics import deception_gain, ei_leverage_table
from .analysis.regression import logit_with_clusters
from .causal.ei import ei_from_do_table
from .codebook import all_ids
from .experiment.incremental_output import IncrementalCsvWriter, PeriodicAnalysisRefresher
from .experiment.runner import trial_to_analysis_row
from .studies.study2_factorial import PRIORITY_INTERACTIONS
from .studies.study3_generalization import _ei_by
from .studies.study4_adversarial import FAKEABLE_DIMS

if TYPE_CHECKING:
    from .data.schema import Article, Query, Trial

_log = logging.getLogger(__name__)


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def model_slug(model: str) -> str:
    """Filesystem-safe directory name for a model roster slug."""
    return (
        model.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .strip() or "unknown_model"
    )


def models_to_save(frame: pd.DataFrame, models: Sequence[str]) -> List[str]:
    """Stable model order: CLI order first, then any extras in the frame."""
    if "model" not in frame.columns:
        return list(models)
    present = set(frame["model"].astype(str))
    out: List[str] = []
    seen: set[str] = set()
    for m in models:
        if m in present and m not in seen:
            out.append(m)
            seen.add(m)
    for m in frame["model"].astype(str).unique():
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def study_model_dir(study: str, model: str, output_root: Path) -> Path:
    return output_root / study / model_slug(model)


def _analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "api_error" in frame.columns:
        # Permanently-rejected trials (e.g. content-filter blocks) carry no valid
        # decision; treat them as missing data rather than y=0.
        frame = frame[frame["api_error"].fillna("").astype(str) == ""]
    if "parse_ok" in frame.columns:
        return frame[frame["parse_ok"] == 1].copy()
    return frame.copy()


def _study1_pair_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["pair_key"] = (
        frame["query_id"].astype(str)
        + "|" + frame["model"].astype(str)
        + "|" + frame["prompt_style"].astype(str)
        + "|" + frame["target_position"].astype(str)
        + "|" + frame["seed"].astype(str)
        + "|" + frame.get("pair_id", "").astype(str)
    )
    return frame


def study1_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = _study1_pair_key(frame)
    analysis_df = _analysis_frame(work)
    ate = ate_table(analysis_df, paired_key="pair_key", cluster="query_id")
    ei = ei_leverage_table(analysis_df, route="experimental", scope_col="target_dim")
    return ate, ei


def study2_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefs = logit_with_clusters(
        frame, factors=all_ids(), interactions=list(PRIORITY_INTERACTIONS), cluster="query_id"
    )
    ei = ei_leverage_table(frame, route="backdoor")
    return coefs, ei


def study3_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df = _analysis_frame(frame)
    ei = ei_leverage_table(analysis_df, route="experimental", scope_col="target_dim")
    ei_by_domain = (
        _ei_by(analysis_df, "domain", scope_col="target_dim")
        if "domain" in analysis_df.columns else pd.DataFrame()
    )
    return ei, ei_by_domain


def study4_tables(
    frame: pd.DataFrame,
    fakeable_dims: Sequence[str] = tuple(FAKEABLE_DIMS),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dec_rows = []
    delta_rows = []
    for dim in fakeable_dims:
        sub = frame[frame.get("fakeable_dim", "") == dim]
        if sub.empty:
            continue
        dec_rows.append(deception_gain(sub, dim))
        gen_df = sub[sub["variant"].isin(["none", "genuine"])]
        fake_df = sub[sub["variant"].isin(["none", "fake"])]

        def _ei_for(d):
            p0 = d[d["variant"] == "none"]["y"].mean()
            p1 = d[d["variant"] != "none"]["y"].mean()
            if pd.isna(p0) or pd.isna(p1):
                return float("nan")
            return ei_from_do_table({0: float(p0), 1: float(p1)}).ei

        ei_gen = _ei_for(gen_df)
        ei_fake = _ei_for(fake_df)
        delta_rows.append({
            "factor": dim,
            "EI_genuine": ei_gen,
            "EI_fake": ei_fake,
            "delta_EI": (ei_fake - ei_gen) if not (pd.isna(ei_fake) or pd.isna(ei_gen)) else float("nan"),
        })
    return pd.DataFrame(dec_rows), pd.DataFrame(delta_rows)


def study5_tables(
    retrieval_frame: pd.DataFrame, gen_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-model Study 5 tables: citation ATE, EI, and end-to-end decomposition.

    `retrieval_frame` is model-independent (stage 1), so it is shared across all
    models; `gen_frame` should be the trials for a single model.
    """
    from .studies.study5_rag import e2e_decomposition

    analysis = _analysis_frame(gen_frame)
    if analysis.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    analysis = analysis.copy()
    analysis["pair_key"] = (
        analysis["query_id"].astype(str)
        + "|" + analysis["model"].astype(str)
        + "|" + analysis["prompt_style"].astype(str)
        + "|" + analysis["target_position"].astype(str)
        + "|" + analysis["seed"].astype(str)
        + "|" + analysis.get("pair_id", "").astype(str)
    )
    ate_cite = ate_table(analysis, outcome="y", paired_key="pair_key", cluster="query_id")
    ei = ei_leverage_table(analysis, route="experimental", scope_col="target_dim")
    e2e = e2e_decomposition(retrieval_frame, gen_frame)
    return ate_cite, ei, e2e


def study6_tables(
    retrieval_frame_model: pd.DataFrame, gen_frame_model: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-model Study 6 tables: retrieval ATE, end-to-end ATE, and EI.

    Study 6 is a real end-to-end RAG: `y` already reflects whether the target was
    finally cited (retrieval + citation combined), so there is no separate
    "cite | in context" channel and no product-form decomposition. The retrieval
    channel (`retrieved`) is kept for reference. Both frames should already be
    sliced to a single model. Returns (ate_retrieved, ate_e2e, ei).
    """
    from .studies.study5_rag import _scoped_ate

    ate_retrieved = (
        _scoped_ate(retrieval_frame_model, outcome="retrieved", pair_key="pair_key")
        if not retrieval_frame_model.empty else pd.DataFrame()
    )
    analysis = _analysis_frame(gen_frame_model)
    if analysis.empty:
        return ate_retrieved, pd.DataFrame(), pd.DataFrame()
    analysis = analysis.copy()
    # Pair control vs treatment on everything identical *except* the target's own
    # feature. `target_position` is deliberately excluded: in the end-to-end RAG
    # it is an outcome of retrieval (a better feature lands the target at an
    # earlier rank, or absent), so pairing on it would prevent any match.
    analysis["pair_key"] = (
        analysis["query_id"].astype(str)
        + "|" + analysis["model"].astype(str)
        + "|" + analysis["prompt_style"].astype(str)
        + "|" + analysis["seed"].astype(str)
        + "|" + analysis.get("pair_id", "").astype(str)
    )
    ate_e2e = ate_table(analysis, outcome="y", paired_key="pair_key", cluster="query_id")
    ei = ei_leverage_table(analysis, route="experimental", scope_col="target_dim")
    return ate_retrieved, ate_e2e, ei


def persist_per_model(
    study: str,
    frame: pd.DataFrame,
    models: Sequence[str],
    output_root: Path,
    save_fn: Callable[[pd.DataFrame, Path], None],
) -> List[Path]:
    """Write one subdirectory per tested model; return saved directories."""
    saved: List[Path] = []
    for model in models_to_save(frame, models):
        sub = frame[frame["model"] == model].copy()
        if sub.empty:
            continue
        out = study_model_dir(study, model, output_root)
        save_fn(sub, out)
        saved.append(out)
    return saved


class StudyModelSink:
    """Route completed trials to per-model CSV files and refresh analysis tables."""

    MIN_PARSE_OK_ROWS = 4

    def __init__(
        self,
        study: str,
        models: Sequence[str],
        output_root: Path,
        *,
        refresh_every: int = 100,
        refresh_sec: float = 300.0,
        resume: bool = False,
    ):
        self.study = study
        self.models = list(models)
        self.output_root = output_root
        self.refresh_every = refresh_every
        self.refresh_sec = refresh_sec
        self.resume = resume
        self._articles_by_id: Dict[str, "Article"] = {}
        self._queries_by_id: Dict[str, "Query"] = {}
        self._retrieval_frame: Optional[pd.DataFrame] = None
        self._retrieval_by_model: Dict[str, pd.DataFrame] = {}
        self._writers: Dict[str, IncrementalCsvWriter] = {}
        self._rewrite_writers: Dict[str, IncrementalCsvWriter] = {}
        self._refreshers: Dict[str, PeriodicAnalysisRefresher] = {}
        self._lock = threading.Lock()

    def bind_context(
        self,
        articles_by_id: Dict[str, "Article"],
        queries_by_id: Dict[str, "Query"],
    ) -> None:
        self._articles_by_id = articles_by_id
        self._queries_by_id = queries_by_id

    def set_retrieval_frame(
        self, frame: pd.DataFrame, *, per_model: bool = False
    ) -> None:
        self._retrieval_frame = frame
        self._retrieval_by_model = {}
        if per_model and not frame.empty and "model" in frame.columns:
            for model in frame["model"].astype(str).unique():
                self._retrieval_by_model[model] = frame[
                    frame["model"].astype(str) == model
                ].copy()

    def on_trial(self, trial: "Trial") -> None:
        row = trial_to_analysis_row(
            trial, self._articles_by_id, self._queries_by_id
        )
        model = str(trial.model)
        self._trial_writer(model).append_row(row)
        self._refresher(model).tick()

    def on_rewrite_row(self, row: dict) -> None:
        model = str(row["model"])
        self._rewrite_writer(model).append_row(row)

    def finalize(self) -> List[Path]:
        saved: List[Path] = []
        for model in self.models:
            out = study_model_dir(self.study, model, self.output_root)
            trials_path = out / "trials.csv"
            if trials_path.exists():
                saved.append(out)
            self._refresh_model(model, final=True)
        return saved

    def completed_row_keys(self, model: str, key_cols: Sequence[str]) -> set:
        """Semantic keys of trials already persisted for `model`.

        Keys are tuples over `key_cols` read from trials.csv (all as strings).
        Unlike a `trial_id`, these columns are always present in the frame, so
        resume matching survives trial-id formula changes and re-runs.
        """
        path = study_model_dir(self.study, model, self.output_root) / "trials.csv"
        if not path.exists():
            return set()
        try:
            df = pd.read_csv(path, dtype=str)
        except (OSError, pd.errors.EmptyDataError):
            return set()
        if not set(key_cols).issubset(df.columns):
            return set()
        df = df[list(key_cols)].fillna("")
        return {tuple(r) for r in df.itertuples(index=False, name=None)}

    def dedupe_trials(self, model: str, key_cols: Sequence[str]) -> int:
        """Drop duplicate trials rows (keep first) by semantic key; self-heals a
        file that a prior broken resume appended duplicates into. Ensures the
        `api_error` column exists so subsequent appends stay column-aligned.
        Returns the number of rows removed."""
        path = study_model_dir(self.study, model, self.output_root) / "trials.csv"
        if not path.exists():
            return 0
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError):
            return 0
        if df.empty or not set(key_cols).issubset(df.columns):
            return 0
        before = len(df)
        df = df.drop_duplicates(subset=list(key_cols), keep="first")
        if "api_error" not in df.columns:
            df["api_error"] = ""
        removed = before - len(df)
        if removed > 0 or "api_error" in df.columns:
            df.to_csv(path, index=False, encoding="utf-8-sig")
        return removed

    def load_rewrites(self, model: str) -> Dict[str, List[str]]:
        """Reload a model's saved query rewrites (query_id -> rewritten list)."""
        path = study_model_dir(self.study, model, self.output_root) / "rewrites.csv"
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError):
            return {}
        if "query_id" not in df.columns or "rewritten_queries" not in df.columns:
            return {}
        out: Dict[str, List[str]] = {}
        for _, r in df.iterrows():
            raw = str(r["rewritten_queries"]) if pd.notna(r["rewritten_queries"]) else ""
            out[str(r["query_id"])] = [s for s in raw.split(" ||| ") if s]
        return out

    def _trial_writer(self, model: str) -> IncrementalCsvWriter:
        with self._lock:
            if model not in self._writers:
                path = study_model_dir(self.study, model, self.output_root) / "trials.csv"
                self._writers[model] = IncrementalCsvWriter(
                    path, truncate_on_init=not self.resume
                )
                self._refresher(model)
            return self._writers[model]

    def _rewrite_writer(self, model: str) -> IncrementalCsvWriter:
        with self._lock:
            if model not in self._rewrite_writers:
                path = study_model_dir(self.study, model, self.output_root) / "rewrites.csv"
                self._rewrite_writers[model] = IncrementalCsvWriter(
                    path, truncate_on_init=not self.resume
                )
            return self._rewrite_writers[model]

    def _refresher(self, model: str) -> PeriodicAnalysisRefresher:
        if model not in self._refreshers:

            def _callback(m=model):
                self._refresh_model(m, final=False)

            self._refreshers[model] = PeriodicAnalysisRefresher(
                _callback,
                refresh_every=self.refresh_every,
                refresh_sec=self.refresh_sec,
            )
        return self._refreshers[model]

    def _refresh_model(self, model: str, *, final: bool) -> None:
        out = study_model_dir(self.study, model, self.output_root)
        writer = self._writers.get(model)
        if writer is not None:
            sub = writer.read_dataframe()
        else:
            trials_path = out / "trials.csv"
            if not trials_path.exists():
                return
            try:
                sub = pd.read_csv(trials_path)
            except Exception:
                return
        if sub.empty:
            return
        if not final and "parse_ok" in sub.columns:
            n_ok = int((sub["parse_ok"] == 1).sum())
            if n_ok < self.MIN_PARSE_OK_ROWS:
                _log.debug(
                    "skip periodic refresh for %s/%s: parse_ok=%d",
                    self.study,
                    model,
                    n_ok,
                )
                return
        try:
            self._write_analysis_tables(model, sub, out, final=final)
        except Exception:
            if final:
                raise
            _log.warning(
                "periodic analysis refresh failed for %s/%s",
                self.study,
                model,
                exc_info=True,
            )

    def _write_analysis_tables(
        self,
        model: str,
        sub: pd.DataFrame,
        out: Path,
        *,
        final: bool,
    ) -> None:
        if self.study == "study1":
            ate, ei = study1_tables(sub)
            save_csv(ate, out / "ate.csv")
            save_csv(ei, out / "ei_leverage.csv")
            if final:
                from .analysis.plots import ate_forest, ei_leverage_bar

                ei_leverage_bar(
                    ei, out / "ei_leverage.png", f"Study 1: EI leverage ({model})"
                )
                ate_forest(ate, out / "ate_forest.png", f"Study 1: ATE by feature ({model})")
        elif self.study == "study2":
            coefs, ei = study2_tables(sub)
            save_csv(coefs, out / "coefficients.csv")
            save_csv(ei, out / "ei_leverage.csv")
            if final:
                from .analysis.plots import ei_leverage_bar

                ei_leverage_bar(
                    ei, out / "ei_leverage.png", f"Study 2: EI leverage ({model})"
                )
        elif self.study == "study3":
            ei, ei_by_domain = study3_tables(sub)
            save_csv(ei, out / "ei_leverage.csv")
            if not ei_by_domain.empty:
                save_csv(ei_by_domain, out / "ei_by_domain.csv")
        elif self.study == "study4":
            deception, delta_ei = study4_tables(sub)
            save_csv(deception, out / "deception.csv")
            save_csv(delta_ei, out / "delta_ei.csv")
        elif self.study == "study_rag":
            retr = self._retrieval_frame if self._retrieval_frame is not None else pd.DataFrame()
            ate_cite, ei, e2e = study5_tables(retr, sub)
            save_csv(ate_cite, out / "ate_cite.csv")
            save_csv(ei, out / "ei_leverage.csv")
            save_csv(e2e, out / "e2e.csv")
        elif self.study == "study6":
            retr = self._retrieval_by_model.get(model, pd.DataFrame())
            ate_retrieved, ate_e2e, ei = study6_tables(retr, sub)
            save_csv(ate_retrieved, out / "ate_retrieved.csv")
            save_csv(ate_e2e, out / "ate_e2e.csv")
            save_csv(ei, out / "ei_leverage.csv")
