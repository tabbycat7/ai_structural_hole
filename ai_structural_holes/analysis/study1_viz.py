"""Interactive and static visualizations for Study 1 outputs."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..codebook import DIMENSIONS, all_ids, get_dimension


FEATURE_ORDER = all_ids()
LAYER_LABELS = {"S": "语义层", "O": "结构层"}
DOMAIN_LABELS = {
    "academic_qa": "学术问答",
    "consumer_product": "消费产品",
    "finance": "金融",
    "health": "健康",
    "travel": "旅行",
}


@dataclass(frozen=True)
class Study1VizResult:
    out_dir: Path
    data_paths: List[Path]
    figure_paths: List[Path]
    html_path: Path


def build_study1_viz(
    study_dir: Path | str = Path("outputs/study1"),
    out_dir: Path | str | None = None,
) -> Study1VizResult:
    """Build Study 1 summary data, static figures, and an interactive HTML report."""
    study_dir = Path(study_dir)
    if out_dir is None:
        out_dir = study_dir / "viz"
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    figures_dir = out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = _study_model_dirs(study_dir)
    if not model_dirs:
        raise FileNotFoundError(f"No Study 1 model outputs found in {study_dir}")

    model_feature, domain_feature, validity, position = _build_tables(model_dirs)
    compass = _build_compass(model_feature)
    model_order_clustered = _cluster_model_order(model_feature)

    data_paths: List[Path] = []
    data_paths.extend(_write_table(model_feature, data_dir / "model_feature_summary"))
    data_paths.extend(_write_table(domain_feature, data_dir / "domain_feature_summary"))
    data_paths.extend(_write_table(validity, data_dir / "validity_summary"))
    data_paths.extend(_write_table(position, data_dir / "position_summary"))
    data_paths.extend(_write_table(compass, data_dir / "feature_compass_summary"))
    data_paths.extend(_write_json_payloads(
        data_dir,
        model_feature=model_feature,
        domain_feature=domain_feature,
        validity=validity,
        position=position,
        compass=compass,
        features=_feature_payload(),
        callouts=_callouts(model_feature, compass, validity),
        model_order_clustered=model_order_clustered,
    ))

    figure_paths = _write_figures(
        model_feature=model_feature,
        domain_feature=domain_feature,
        validity=validity,
        position=position,
        compass=compass,
        model_order=model_order_clustered,
        out_dir=figures_dir,
    )
    html_path = _write_html(
        out_dir / "index.html",
        model_feature=model_feature,
        domain_feature=domain_feature,
        validity=validity,
        position=position,
        compass=compass,
        model_order_clustered=model_order_clustered,
        figures=[p.relative_to(out_dir).as_posix() for p in figure_paths],
    )
    return Study1VizResult(out_dir=out_dir, data_paths=data_paths, figure_paths=figure_paths, html_path=html_path)


def _study_model_dirs(study_dir: Path) -> List[Path]:
    dirs = []
    for d in sorted(study_dir.iterdir()):
        if d.is_dir() and (d / "trials.csv").exists() and (d / "ate.csv").exists():
            dirs.append(d)
    return dirs


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _write_table(df: pd.DataFrame, stem: Path) -> List[Path]:
    csv_path = stem.with_suffix(".csv")
    json_path = stem.with_suffix(".json")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(_json_records(df), encoding="utf-8")
    return [csv_path, json_path]


def _write_json_payloads(data_dir: Path, **payload) -> List[Path]:
    path = data_dir / "study1_viz_payload.json"
    serial = {k: _jsonable(v) for k, v in payload.items()}
    serial["generated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")
    return [path]


def _json_records(df: pd.DataFrame) -> str:
    return json.dumps(_jsonable(df), ensure_ascii=False, indent=2)


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(r) for r in obj.replace({np.nan: None}).to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    return obj


def _build_tables(model_dirs: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mf_rows: List[dict] = []
    domain_rows: List[dict] = []
    validity_rows: List[dict] = []
    position_rows: List[dict] = []

    for model_dir in model_dirs:
        trials = _prepare_trials(_read_csv(model_dir / "trials.csv"), model_dir.name)
        model = _first_value(trials, "model", model_dir.name.replace("__", "/"))
        display = _display_model_name(model, model_dir.name)

        ate = _normalize_factor_col(_read_csv(model_dir / "ate.csv"))
        ei = _normalize_factor_col(_read_csv(model_dir / "ei_leverage.csv")) if (model_dir / "ei_leverage.csv").exists() else pd.DataFrame()
        ei = _prepare_ei(ei)
        parse_rate = float(trials["parse_ok_bool"].mean()) if len(trials) else float("nan")

        for _, row in ate.iterrows():
            factor = str(row["factor"])
            dim = get_dimension(factor)
            ei_row = ei[ei["factor"] == factor]
            ei_value = _float(ei_row["EI"].iloc[0]) if len(ei_row) else 0.0
            ei_norm = _float(ei_row["EI_norm"].iloc[0]) if len(ei_row) else 0.0
            ei_share = _float(ei_row["EI_share"].iloc[0]) if len(ei_row) and "EI_share" in ei_row else 0.0
            ei_rank = int(ei_row["EI_rank"].iloc[0]) if len(ei_row) and "EI_rank" in ei_row else None
            ate_value = _float(row.get("ATE"))
            ci_low = _float(row.get("ci_low"))
            ci_high = _float(row.get("ci_high"))
            sig = bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
            mf_rows.append({
                "model_slug": model_dir.name,
                "model": model,
                "model_display": display,
                "factor": factor,
                "feature_name": dim.name,
                "feature_label": f"{factor} {dim.name.split(' (')[0]}",
                "layer": dim.layer.value,
                "layer_label": LAYER_LABELS.get(dim.layer.value, dim.layer.value),
                "ATE": ate_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_treated": int(row.get("n_treated", 0)),
                "n_control": int(row.get("n_control", 0)),
                "level_treated": int(row.get("level_treated", dim.top_code())),
                "level_control": int(row.get("level_control", dim.baseline_code())),
                "EI": ei_value,
                "EI_norm": ei_norm,
                "EI_share": ei_share,
                "EI_rank": ei_rank,
                "significant": sig,
                "direction": _direction(ate_value, ci_low, ci_high),
                "parse_ok_rate": parse_rate,
            })

        analysis = trials[trials["parse_ok_bool"]].copy()
        domain_rows.extend(_domain_feature_rows(analysis, model_dir.name, model, display))
        validity_rows.append(_validity_row(trials, model_dir.name, model, display))
        position_rows.extend(_position_rows(trials, model_dir.name, model, display))

    model_feature = pd.DataFrame(mf_rows)
    if len(model_feature):
        model_feature["feature_order"] = model_feature["factor"].map({f: i for i, f in enumerate(FEATURE_ORDER)})
        model_feature["abs_ATE"] = model_feature["ATE"].abs()
        max_abs = float(model_feature["ATE"].abs().max()) or 1.0
        max_ei = float(model_feature["EI"].max()) or 1.0
        model_feature["ATE_scaled"] = model_feature["ATE"] / max_abs
        model_feature["EI_scaled"] = model_feature["EI"] / max_ei
        model_feature = model_feature.sort_values(["model_display", "feature_order"]).reset_index(drop=True)

    domain_feature = pd.DataFrame(domain_rows)
    if len(domain_feature):
        domain_feature["domain_label"] = domain_feature["domain"].map(DOMAIN_LABELS).fillna(domain_feature["domain"])
        domain_feature["feature_order"] = domain_feature["factor"].map({f: i for i, f in enumerate(FEATURE_ORDER)})
        domain_feature = domain_feature.sort_values(["domain", "model_display", "feature_order"]).reset_index(drop=True)

    validity = pd.DataFrame(validity_rows).sort_values("model_display").reset_index(drop=True)
    position = pd.DataFrame(position_rows).sort_values(["model_display", "target_position"]).reset_index(drop=True)
    return model_feature, domain_feature, validity, position


def _normalize_factor_col(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: c.lstrip("\ufeff") for c in df.columns}
    return df.rename(columns=rename)


def _prepare_trials(df: pd.DataFrame, model_slug: str) -> pd.DataFrame:
    df = _normalize_factor_col(df).copy()
    if "model" not in df.columns:
        df["model"] = model_slug.replace("__", "/")
    for col in FEATURE_ORDER:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["parse_ok_bool"] = _bool_series(df.get("parse_ok", pd.Series([1] * len(df))))
    df["y_num"] = pd.to_numeric(df.get("y", pd.Series([np.nan] * len(df))), errors="coerce")
    if "pair_key" not in df.columns or df["pair_key"].isna().all() or (df["pair_key"].astype(str).str.len() == 0).all():
        df["pair_key"] = _rebuild_pair_key(df)
    else:
        missing = df["pair_key"].isna() | (df["pair_key"].astype(str).str.len() == 0)
        if missing.any():
            df.loc[missing, "pair_key"] = _rebuild_pair_key(df.loc[missing])
    return df


def _rebuild_pair_key(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["query_id", "model", "prompt_style", "target_position", "seed", "pair_id"]:
        if col in df.columns:
            parts.append(df[col].astype(str))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    out = parts[0]
    for part in parts[1:]:
        out = out + "|" + part
    return out


def _prepare_ei(ei: pd.DataFrame) -> pd.DataFrame:
    if ei.empty:
        return ei
    ei = ei.copy()
    for col in ["EI", "EI_norm", "ATE", "ate_ci_low", "ate_ci_high", "EI_share"]:
        if col in ei.columns:
            ei[col] = pd.to_numeric(ei[col], errors="coerce")
    if "EI_share" not in ei.columns:
        total = float(ei["EI"].sum()) if "EI" in ei.columns else 0.0
        ei["EI_share"] = ei["EI"] / total if total > 0 else 0.0
    ei = ei.sort_values("EI", ascending=False).reset_index(drop=True)
    ei["EI_rank"] = np.arange(1, len(ei) + 1)
    return ei


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1", "1.0", "true", "yes"})


def _first_value(df: pd.DataFrame, col: str, default: str) -> str:
    if col not in df.columns or df.empty:
        return default
    vals = df[col].dropna().astype(str)
    return vals.iloc[0] if len(vals) else default


def _float(value) -> float:
    try:
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        out = float(value)
        return out if not math.isnan(out) else 0.0
    except Exception:
        return 0.0


def _direction(ate: float, ci_low: float, ci_high: float) -> str:
    if ci_low > 0 and ci_high > 0:
        return "显著吸引"
    if ci_low < 0 and ci_high < 0:
        return "显著排斥"
    if ate > 0:
        return "偏吸引"
    if ate < 0:
        return "偏排斥"
    return "中性"


def _display_model_name(model: str, slug: str) -> str:
    raw = (model or slug.replace("__", "/")).split("/")[-1]
    raw = raw.replace("__", "/").replace("_", " ").replace("-", " ")
    aliases = {
        "gpt 5.4 mini": "GPT-5.4 mini",
        "claude haiku 4.5": "Claude Haiku 4.5",
        "llama 4 maverick": "Llama 4 Maverick",
        "deepseek v4 flash": "DeepSeek V4 Flash",
        "mistral small 2603": "Mistral Small 2603",
        "kimi k2.5": "Kimi K2.5",
        "gemini 3.1 flash lite": "Gemini 3.1 Flash Lite",
        "qwen3.6 flash": "Qwen 3.6 Flash",
        "MiniMax M3": "MiniMax M3",
        "doubao seed 2 0 mini 260428": "Doubao Seed 2.0 Mini",
    }
    lowered = raw.strip().lower()
    for k, v in aliases.items():
        if lowered == k.lower():
            return v
    if "MiniMax" in raw or "minimax" in raw.lower():
        return "MiniMax M3" if "M3" in raw else raw
    return " ".join(part.capitalize() if not part.isupper() else part for part in raw.split())


def _domain_feature_rows(df: pd.DataFrame, model_slug: str, model: str, display: str) -> List[dict]:
    rows = []
    if df.empty or "domain" not in df.columns:
        return rows
    domains = sorted(df["domain"].dropna().astype(str).unique().tolist())
    for factor in FEATURE_ORDER:
        if factor not in df.columns:
            continue
        dim = get_dimension(factor)
        lo, hi = dim.baseline_code(), dim.top_code()
        sub = df.loc[df[factor].isin([lo, hi]), ["domain", "pair_key", factor, "y_num"]].dropna(subset=["y_num"])
        if sub.empty:
            grouped = pd.DataFrame()
            mean_piv = pd.DataFrame()
            count_piv = pd.DataFrame()
        else:
            grouped = (
                sub.groupby(["domain", "pair_key", factor], sort=False)["y_num"]
                .agg(["mean", "count"])
                .reset_index()
            )
            mean_piv = grouped.pivot_table(
                index=["domain", "pair_key"], columns=factor, values="mean", aggfunc="first"
            )
            count_piv = grouped.pivot_table(
                index=["domain", "pair_key"], columns=factor, values="count", aggfunc="sum"
            )
        lo_col = _pivot_level_col(mean_piv, lo)
        hi_col = _pivot_level_col(mean_piv, hi)
        if lo_col is not None and hi_col is not None:
            diff_frame = pd.DataFrame({
                "ATE_pair": mean_piv[hi_col] - mean_piv[lo_col],
                "n_treated_pair": count_piv.get(hi_col, 0),
                "n_control_pair": count_piv.get(lo_col, 0),
            }).reset_index()
            diff_frame = diff_frame.dropna(subset=["ATE_pair"])
        else:
            diff_frame = pd.DataFrame(columns=["domain", "pair_key", "ATE_pair", "n_treated_pair", "n_control_pair"])

        if not diff_frame.empty:
            by_domain = diff_frame.groupby("domain").agg(
                ATE=("ATE_pair", "mean"),
                n_pairs=("ATE_pair", "size"),
                n_treated=("n_treated_pair", "sum"),
                n_control=("n_control_pair", "sum"),
            )
        else:
            by_domain = pd.DataFrame(columns=["ATE", "n_pairs", "n_treated", "n_control"])

        for domain in domains:
            stats = by_domain.loc[domain] if domain in by_domain.index else None
            ate = float(stats["ATE"]) if stats is not None else 0.0
            n_pairs = int(stats["n_pairs"]) if stats is not None else 0
            n_treated = int(stats["n_treated"]) if stats is not None else 0
            n_control = int(stats["n_control"]) if stats is not None else 0
            rows.append({
                "model_slug": model_slug,
                "model": model,
                "model_display": display,
                "domain": str(domain),
                "factor": factor,
                "feature_name": dim.name,
                "layer": dim.layer.value,
                "ATE": ate,
                "n_pairs": n_pairs,
                "n_treated": n_treated,
                "n_control": n_control,
            })
    return rows


def _pivot_level_col(frame: pd.DataFrame, level: int):
    if frame.empty:
        return None
    if level in frame.columns:
        return level
    as_float = float(level)
    if as_float in frame.columns:
        return as_float
    as_str = str(level)
    if as_str in frame.columns:
        return as_str
    return None


def _validity_row(df: pd.DataFrame, model_slug: str, model: str, display: str) -> dict:
    ok = df["parse_ok_bool"]
    y_ok = df.loc[ok, "y_num"]
    row = {
        "model_slug": model_slug,
        "model": model,
        "model_display": display,
        "n_trials": int(len(df)),
        "parse_ok_n": int(ok.sum()),
        "parse_ok_rate": float(ok.mean()) if len(df) else 0.0,
        "selection_rate": float(y_ok.mean()) if len(y_ok) else 0.0,
        "n_domains": int(df["domain"].nunique()) if "domain" in df else 0,
        "n_features": int(df["target_dim"].nunique()) if "target_dim" in df else 0,
        "n_positions": int(df["target_position"].nunique()) if "target_position" in df else 0,
        "n_seeds": int(df["seed"].nunique()) if "seed" in df else 0,
    }
    for pos, rate in _position_rate_map(df).items():
        row[f"position_{pos}_selection_rate"] = rate
    return row


def _position_rows(df: pd.DataFrame, model_slug: str, model: str, display: str) -> List[dict]:
    rows = []
    if "target_position" not in df.columns:
        return rows
    for pos, g in df.groupby("target_position"):
        ok = g["parse_ok_bool"]
        y_ok = g.loc[ok, "y_num"]
        rows.append({
            "model_slug": model_slug,
            "model": model,
            "model_display": display,
            "target_position": int(pos),
            "n_trials": int(len(g)),
            "parse_ok_rate": float(ok.mean()) if len(g) else 0.0,
            "selection_rate": float(y_ok.mean()) if len(y_ok) else 0.0,
        })
    return rows


def _position_rate_map(df: pd.DataFrame) -> Dict[int, float]:
    out = {}
    if "target_position" not in df.columns:
        return out
    for pos, g in df.groupby("target_position"):
        ok = g["parse_ok_bool"]
        vals = g.loc[ok, "y_num"]
        out[int(pos)] = float(vals.mean()) if len(vals) else 0.0
    return out


def _build_compass(model_feature: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FEATURE_ORDER:
        sub = model_feature[model_feature["factor"] == factor]
        if sub.empty:
            continue
        dim = get_dimension(factor)
        ate = sub["ATE"].astype(float)
        ei = sub["EI"].astype(float)
        rows.append({
            "factor": factor,
            "feature_name": dim.name,
            "feature_label": f"{factor} {dim.name.split(' (')[0]}",
            "layer": dim.layer.value,
            "mean_ATE": float(ate.mean()),
            "median_ATE": float(ate.median()),
            "sd_ATE": float(ate.std(ddof=0)),
            "min_ATE": float(ate.min()),
            "max_ATE": float(ate.max()),
            "mean_EI": float(ei.mean()),
            "median_EI": float(ei.median()),
            "top1_count": int((sub["EI_rank"] == 1).sum()) if "EI_rank" in sub else 0,
            "significant_positive_models": int((sub["ci_low"] > 0).sum()),
            "significant_negative_models": int((sub["ci_high"] < 0).sum()),
            "positive_models": int((ate > 0).sum()),
            "negative_models": int((ate < 0).sum()),
        })
    return pd.DataFrame(rows)


def _cluster_model_order(model_feature: pd.DataFrame) -> List[str]:
    pivot = (
        model_feature.pivot(index="model_display", columns="factor", values="ATE")
        .reindex(columns=FEATURE_ORDER)
        .fillna(0.0)
    )
    if len(pivot) <= 2:
        return pivot.index.tolist()
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        dist = pdist(pivot.values, metric="correlation")
        if np.isnan(dist).any() or np.allclose(dist, 0):
            dist = pdist(pivot.values, metric="euclidean")
        order = leaves_list(linkage(dist, method="average"))
        return pivot.index[order].tolist()
    except Exception:
        return pivot.assign(_s1=pivot.get("S1", 0)).sort_values("_s1", ascending=False).index.tolist()


def _write_figures(
    *,
    model_feature: pd.DataFrame,
    domain_feature: pd.DataFrame,
    validity: pd.DataFrame,
    position: pd.DataFrame,
    compass: pd.DataFrame,
    model_order: Sequence[str],
    out_dir: Path,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    _set_matplotlib_style(plt)
    paths = [
        _fig_bias_map(model_feature, model_order, out_dir / "model_feature_bias_map.png", plt, colors),
        _fig_constellation(model_feature, model_order, out_dir / "structural_hole_constellation.png", plt),
        _fig_fingerprints(model_feature, model_order, out_dir / "model_fingerprints.png", plt),
        _fig_compass(compass, out_dir / "leverage_compass.png", plt),
        _fig_domain_prism(domain_feature, out_dir / "domain_prism.png", plt, colors),
        _fig_validity(validity, position, out_dir / "validity_strips.png", plt),
    ]
    return paths


def _set_matplotlib_style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": "#08111f",
        "axes.facecolor": "#0b1628",
        "axes.edgecolor": "#2f4261",
        "axes.labelcolor": "#d8e6ff",
        "xtick.color": "#9eb4d8",
        "ytick.color": "#9eb4d8",
        "text.color": "#eaf2ff",
        "font.family": "sans-serif",
        "font.sans-serif": [
            "PingFang SC",
            "Arial Unicode MS",
            "Noto Sans CJK SC",
            "Microsoft YaHei",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "savefig.facecolor": "#08111f",
    })


def _signed_cmap(colors):
    return colors.LinearSegmentedColormap.from_list(
        "study1_signed",
        ["#4cc9f0", "#18243a", "#f7f7ff", "#ffb703", "#ff4d6d"],
        N=256,
    )


def _fig_bias_map(df: pd.DataFrame, model_order: Sequence[str], path: Path, plt, colors) -> Path:
    pivot = df.pivot(index="model_display", columns="factor", values="ATE").reindex(index=model_order, columns=FEATURE_ORDER)
    ei = df.pivot(index="model_display", columns="factor", values="EI").reindex(index=model_order, columns=FEATURE_ORDER).fillna(0)
    sig = df.pivot(index="model_display", columns="factor", values="significant").reindex(index=model_order, columns=FEATURE_ORDER).fillna(False)
    fig, ax = plt.subplots(figsize=(12, 7))
    vmax = max(0.01, float(np.nanmax(np.abs(pivot.values))))
    im = ax.imshow(pivot.values, cmap=_signed_cmap(colors), vmin=-vmax, vmax=vmax, aspect="auto")
    max_ei = max(1e-9, float(ei.values.max()))
    for y, model in enumerate(pivot.index):
        for x, factor in enumerate(pivot.columns):
            size = 40 + 520 * math.sqrt(float(ei.loc[model, factor]) / max_ei)
            edge = "#f7f7ff" if bool(sig.loc[model, factor]) else "#5d7198"
            ax.scatter(x, y, s=size, facecolors="none", edgecolors=edge, linewidths=1.4)
            val = pivot.loc[model, factor]
            ax.text(x, y, f"{val:+.2f}", ha="center", va="center", fontsize=8, color="#06101d")
    ax.set_xticks(np.arange(len(FEATURE_ORDER)))
    ax.set_xticklabels(FEATURE_ORDER, fontsize=11)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title("Study 1 Model × Feature Bias Map: ATE color, EI ring size", fontsize=15, pad=16)
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("ATE (selection-rate difference)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _fig_constellation(df: pd.DataFrame, model_order: Sequence[str], path: Path, plt) -> Path:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.axis("off")
    max_ei = max(1e-9, float(df["EI"].max()))
    max_abs = max(0.01, float(df["ATE"].abs().max()))
    radii = {m: 1.5 + i * 0.12 for i, m in enumerate(model_order)}
    angles = {f: -math.pi / 2 + i * 2 * math.pi / len(FEATURE_ORDER) for i, f in enumerate(FEATURE_ORDER)}
    for f, ang in angles.items():
        ax.plot([0, math.cos(ang) * 2.95], [0, math.sin(ang) * 2.95], color="#273955", lw=1)
        ax.text(math.cos(ang) * 3.18, math.sin(ang) * 3.18, f, ha="center", va="center", fontsize=13, weight="bold")
    for r in radii.values():
        circ = plt.Circle((0, 0), r, edgecolor="#172842", facecolor="none", lw=0.55, alpha=0.55)
        ax.add_patch(circ)
    for _, row in df.iterrows():
        ang = angles[row["factor"]]
        r = radii.get(row["model_display"], 1.5)
        x, y = math.cos(ang) * r, math.sin(ang) * r
        color = _signed_color(row["ATE"], max_abs)
        size = 28 + 210 * math.sqrt(float(row["EI"]) / max_ei)
        ax.scatter([x], [y], s=size, color=color, edgecolors="#f7f7ff" if row["significant"] else "#38506f", linewidths=1.0, alpha=0.95)
    ax.text(0, 0.05, "Study 1\nPreference\nConstellation", ha="center", va="center", fontsize=16, weight="bold")
    ax.set_xlim(-3.45, 3.45)
    ax.set_ylim(-3.45, 3.45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _fig_fingerprints(df: pd.DataFrame, model_order: Sequence[str], path: Path, plt) -> Path:
    n = len(model_order)
    cols = 5
    rows = int(math.ceil(n / cols))
    angles = np.linspace(0, 2 * np.pi, len(FEATURE_ORDER), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    max_abs = max(0.05, float(df["ATE"].abs().max()))
    fig, axes = plt.subplots(rows, cols, subplot_kw={"projection": "polar"}, figsize=(15, 3.4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, model in zip(axes, model_order):
        vals = (
            df[df["model_display"] == model]
            .set_index("factor")
            .reindex(FEATURE_ORDER)["ATE"]
            .fillna(0)
            .values
        )
        closed_vals = np.r_[vals, vals[0]]
        ax.plot(closed_angles, closed_vals, color="#8bd3ff", lw=1.8)
        ax.fill(closed_angles, np.maximum(closed_vals, 0), color="#ffb703", alpha=0.22)
        ax.fill(closed_angles, np.minimum(closed_vals, 0), color="#4cc9f0", alpha=0.18)
        ax.set_ylim(-max_abs, max_abs)
        ax.set_xticks(angles)
        ax.set_xticklabels(FEATURE_ORDER, fontsize=8)
        ax.set_yticklabels([])
        ax.grid(color="#263b5b", alpha=0.75)
        ax.set_title(model, fontsize=10, pad=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Model Preference Fingerprints (ATE around 8 features)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _fig_compass(df: pd.DataFrame, path: Path, plt) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    max_ei = max(1e-9, float(df["mean_EI"].max()))
    colors = ["#ffb703" if v >= 0 else "#4cc9f0" for v in df["mean_ATE"]]
    sizes = 220 + 1800 * np.sqrt(df["mean_EI"] / max_ei)
    ax.scatter(df["mean_ATE"], df["sd_ATE"], s=sizes, c=colors, edgecolors="#f7f7ff", linewidths=1.1, alpha=0.9)
    for _, row in df.iterrows():
        ax.text(row["mean_ATE"], row["sd_ATE"] + 0.004, row["factor"], ha="center", va="bottom", fontsize=12, weight="bold")
    ax.axvline(0, color="#7d8fb1", lw=1, ls="--")
    ax.set_xlabel("Mean ATE across models")
    ax.set_ylabel("Model disagreement (SD of ATE)")
    ax.set_title("Leverage Compass: agreement, direction, and EI", fontsize=15)
    ax.grid(color="#263b5b", alpha=0.65)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _fig_domain_prism(df: pd.DataFrame, path: Path, plt, colors) -> Path:
    summary = (
        df.groupby(["domain", "factor"], as_index=False)["ATE"].mean()
        .pivot(index="domain", columns="factor", values="ATE")
        .reindex(columns=FEATURE_ORDER)
    )
    summary = summary.reindex([d for d in DOMAIN_LABELS if d in summary.index])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    vmax = max(0.01, float(np.nanmax(np.abs(summary.values))))
    im = ax.imshow(summary.values, cmap=_signed_cmap(colors), vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(FEATURE_ORDER)))
    ax.set_xticklabels(FEATURE_ORDER, fontsize=11)
    ax.set_yticks(np.arange(len(summary.index)))
    ax.set_yticklabels([DOMAIN_LABELS.get(d, d) for d in summary.index], fontsize=11)
    for y, dom in enumerate(summary.index):
        for x, factor in enumerate(summary.columns):
            ax.text(x, y, f"{summary.loc[dom, factor]:+.2f}", ha="center", va="center", fontsize=8, color="#06101d")
    ax.set_title("Domain Prism: mean paired ATE by domain and feature", fontsize=15, pad=14)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Mean ATE across models")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _fig_validity(validity: pd.DataFrame, position: pd.DataFrame, path: Path, plt) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    val = validity.sort_values("parse_ok_rate")
    axes[0].barh(val["model_display"], val["parse_ok_rate"], color="#8bd3ff")
    axes[0].set_xlim(0.94, 1.002)
    axes[0].set_xlabel("Parse OK rate")
    axes[0].set_title("Parsing reliability")
    for i, (_, row) in enumerate(val.iterrows()):
        axes[0].text(row["parse_ok_rate"] + 0.001, i, f"{row['parse_ok_rate']:.3f}", va="center", fontsize=8)

    pos_pivot = position.pivot(index="model_display", columns="target_position", values="selection_rate").reindex(index=validity["model_display"])
    x = np.arange(len(pos_pivot.index))
    width = 0.24
    colors = ["#4cc9f0", "#ffb703", "#ff4d6d"]
    for i, pos in enumerate(pos_pivot.columns):
        axes[1].bar(x + (i - 1) * width, pos_pivot[pos], width=width, label=f"pos {pos}", color=colors[i % len(colors)], alpha=0.82)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pos_pivot.index, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Selection rate")
    axes[1].set_title("Position-balance diagnostic")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="x", color="#263b5b", alpha=0.55)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _signed_color(value: float, max_abs: float) -> str:
    t = max(-1.0, min(1.0, float(value) / max_abs))
    neg = np.array([0x4C, 0xC9, 0xF0])
    mid = np.array([0xF7, 0xF7, 0xFF])
    pos = np.array([0xFF, 0x4D, 0x6D])
    if t >= 0:
        c = mid * (1 - t) + pos * t
    else:
        c = mid * (1 + t) + neg * (-t)
    return "#" + "".join(f"{int(round(v)):02x}" for v in c)


def _write_html(
    path: Path,
    *,
    model_feature: pd.DataFrame,
    domain_feature: pd.DataFrame,
    validity: pd.DataFrame,
    position: pd.DataFrame,
    compass: pd.DataFrame,
    model_order_clustered: Sequence[str],
    figures: Sequence[str],
) -> Path:
    data = {
        "modelFeature": _jsonable(model_feature),
        "domainFeature": _jsonable(domain_feature),
        "validity": _jsonable(validity),
        "position": _jsonable(position),
        "compass": _jsonable(compass),
        "features": _feature_payload(),
        "modelOrderClustered": list(model_order_clustered),
        "modelOrderOriginal": sorted(model_feature["model_display"].unique().tolist()),
        "domains": [{"id": k, "label": v} for k, v in DOMAIN_LABELS.items()],
        "figures": list(figures),
        "callouts": _callouts(model_feature, compass, validity),
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
    }
    html = HTML_TEMPLATE.replace("__STUDY1_DATA__", json.dumps(data, ensure_ascii=False))
    path.write_text(html, encoding="utf-8")
    return path


def _feature_payload() -> List[dict]:
    rows = []
    for i, factor in enumerate(FEATURE_ORDER):
        dim = DIMENSIONS[factor]
        rows.append({
            "id": factor,
            "order": i,
            "name": dim.name,
            "label": f"{factor} {dim.name.split(' (')[0]}",
            "layer": dim.layer.value,
            "layerLabel": LAYER_LABELS.get(dim.layer.value, dim.layer.value),
            "definition": dim.definition,
        })
    return rows


def _callouts(model_feature: pd.DataFrame, compass: pd.DataFrame, validity: pd.DataFrame) -> dict:
    s1 = model_feature[model_feature["factor"] == "S1"].copy()
    if len(s1):
        top = s1.loc[s1["ATE"].idxmax()]
        bottom = s1.loc[s1["ATE"].idxmin()]
        s1_text = (
            f"S1 最大反转: {top['model_display']} {top['ATE']:+.1%} vs "
            f"{bottom['model_display']} {bottom['ATE']:+.1%}"
        )
    else:
        s1_text = "S1 反转: 数据不足"
    top_feature = compass.sort_values("mean_EI", ascending=False).iloc[0] if len(compass) else None
    parse_min = validity.sort_values("parse_ok_rate").iloc[0] if len(validity) else None
    return {
        "models": int(model_feature["model_display"].nunique()),
        "features": int(model_feature["factor"].nunique()),
        "trials": int(validity["n_trials"].sum()) if len(validity) else 0,
        "s1Reversal": s1_text,
        "topLeverage": (
            f"平均 EI 最高: {top_feature['factor']} ({top_feature['mean_EI']:.4f})"
            if top_feature is not None else "平均 EI: 数据不足"
        ),
        "parseFloor": (
            f"最低解析率: {parse_min['model_display']} {parse_min['parse_ok_rate']:.1%}"
            if parse_min is not None else "解析率: 数据不足"
        ),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Study 1 模型偏好星图</title>
  <style>
    :root {
      --bg: #06101d;
      --panel: #0b1628;
      --panel-2: #0f1e34;
      --ink: #eaf2ff;
      --muted: #9eb4d8;
      --line: #263b5b;
      --cold: #4cc9f0;
      --warm: #ff4d6d;
      --gold: #ffb703;
      --white: #f7f7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 30% 10%, rgba(76, 201, 240, 0.15), transparent 28rem),
        radial-gradient(circle at 75% 22%, rgba(255, 183, 3, 0.11), transparent 24rem),
        linear-gradient(180deg, #06101d 0%, #08111f 48%, #050b14 100%);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
      letter-spacing: 0;
    }
    main { width: min(1440px, calc(100vw - 32px)); margin: 0 auto; }
    header {
      min-height: 92vh;
      display: grid;
      grid-template-columns: minmax(300px, 0.86fr) minmax(520px, 1.14fr);
      gap: 28px;
      align-items: center;
      padding: 42px 0 18px;
    }
    h1 { margin: 0; font-size: clamp(38px, 6vw, 92px); line-height: 0.95; letter-spacing: 0; }
    h2 { margin: 0 0 14px; font-size: 24px; }
    h3 { margin: 0 0 10px; font-size: 16px; color: var(--white); }
    p { color: var(--muted); line-height: 1.72; margin: 12px 0; }
    .eyebrow { color: var(--gold); font-size: 13px; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 16px; }
    .hero-copy { max-width: 620px; }
    .stat-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 28px; }
    .stat {
      border: 1px solid var(--line);
      background: rgba(11, 22, 40, 0.72);
      padding: 14px;
      border-radius: 8px;
      min-height: 84px;
    }
    .stat strong { display: block; font-size: 24px; color: var(--white); }
    .stat span { color: var(--muted); font-size: 12px; }
    section { padding: 54px 0; border-top: 1px solid rgba(158, 180, 216, 0.18); }
    .section-head { display: flex; justify-content: space-between; gap: 24px; align-items: end; margin-bottom: 18px; }
    .section-head p { max-width: 780px; margin-bottom: 0; }
    .viz-panel {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(11, 22, 40, 0.92), rgba(8, 17, 31, 0.92));
      border-radius: 8px;
      padding: 18px;
      overflow: hidden;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
    }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 14px; align-items: center; }
    button, select {
      color: var(--ink);
      background: #13233b;
      border: 1px solid #334a6f;
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }
    button.active { border-color: var(--gold); color: var(--gold); }
    svg { width: 100%; height: auto; display: block; }
    .axis text, .tick text { fill: var(--muted); font-size: 12px; }
    .grid-line { stroke: var(--line); stroke-width: 1; }
    .feature-label { fill: var(--white); font-size: 13px; font-weight: 700; }
    .model-label { fill: var(--muted); font-size: 12px; }
    .small { color: var(--muted); font-size: 12px; }
    .callout-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }
    .callout {
      border-left: 3px solid var(--gold);
      background: rgba(15, 30, 52, 0.72);
      padding: 14px;
      min-height: 82px;
      border-radius: 6px;
    }
    .callout b { display: block; margin-bottom: 6px; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .figure-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .figure-grid img {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    table { width: 100%; border-collapse: collapse; color: var(--ink); }
    th, td { border-bottom: 1px solid rgba(158, 180, 216, 0.18); padding: 9px 8px; text-align: left; font-size: 13px; }
    th { color: var(--muted); font-weight: 600; }
    .bar-bg { height: 9px; background: #14243c; border-radius: 999px; overflow: hidden; }
    .bar { height: 100%; background: linear-gradient(90deg, var(--cold), var(--gold)); }
    .tooltip {
      position: fixed;
      pointer-events: none;
      z-index: 10;
      min-width: 220px;
      max-width: 320px;
      padding: 11px 12px;
      border: 1px solid #49658d;
      background: rgba(5, 11, 20, 0.95);
      color: var(--ink);
      border-radius: 8px;
      box-shadow: 0 12px 34px rgba(0,0,0,0.38);
      transform: translate(12px, 12px);
      opacity: 0;
      transition: opacity 120ms ease;
      font-size: 12px;
      line-height: 1.5;
    }
    .tooltip.visible { opacity: 1; }
    .legend { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; color: var(--muted); font-size: 12px; }
    .legend .swatch { width: 34px; height: 10px; border-radius: 999px; display: inline-block; vertical-align: middle; margin-right: 6px; }
    .mono { font-variant-numeric: tabular-nums; }
    @media (max-width: 980px) {
      main { width: min(100vw - 20px, 760px); }
      header, .two-col, .figure-grid { grid-template-columns: 1fr; }
      header { min-height: auto; padding-top: 30px; }
      .stat-row, .callout-grid { grid-template-columns: 1fr; }
      .section-head { display: block; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="hero-copy">
        <div class="eyebrow">Study 1 / Structural-Hole Preference Atlas</div>
        <h1>模型偏好星图</h1>
        <p>10 个被测模型、8 个文章特征、360,000 次选择实验。这里展示的不是排行榜，而是每个模型会被哪些“结构洞线索”牵引、排斥，或者出现阵营分裂。</p>
        <div class="stat-row">
          <div class="stat"><strong id="stat-models">10</strong><span>models</span></div>
          <div class="stat"><strong id="stat-features">8</strong><span>features</span></div>
          <div class="stat"><strong id="stat-trials">360k</strong><span>trials</span></div>
        </div>
      </div>
      <div class="viz-panel">
        <h2>Structural-Hole Constellation</h2>
        <p class="small">每条射线是一个特征；同一射线上 10 个节点是不同模型。颜色表示 ATE 方向，光圈大小表示 EI，亮边表示置信区间不跨 0。</p>
        <div id="constellation"></div>
        <div class="legend">
          <span><i class="swatch" style="background:linear-gradient(90deg,#4cc9f0,#f7f7ff,#ff4d6d)"></i>ATE direction</span>
          <span>circle size = EI</span>
          <span>bright outline = significant</span>
        </div>
      </div>
    </header>

    <section>
      <div class="section-head">
        <div>
          <h2>Model × Feature Bias Map</h2>
          <p>热力颜色是配对 ATE，圆点大小是 EI。默认按模型偏好相似度聚类，也可以切回原始模型顺序。</p>
        </div>
        <div class="toolbar">
          <button id="sort-cluster" class="active">cluster</button>
          <button id="sort-original">original</button>
          <select id="feature-filter" aria-label="feature layer">
            <option value="all">all features</option>
            <option value="S">semantic S</option>
            <option value="O">structural O</option>
          </select>
        </div>
      </div>
      <div class="viz-panel" id="heatmap"></div>
      <div class="callout-grid">
        <div class="callout"><b>反转信号</b><span id="callout-s1"></span></div>
        <div class="callout"><b>最高杠杆</b><span id="callout-ei"></span></div>
        <div class="callout"><b>解析质检</b><span id="callout-parse"></span></div>
      </div>
    </section>

    <section class="two-col">
      <div>
        <h2>Leverage Compass</h2>
        <p>横轴是跨模型平均 ATE，纵轴是模型间分歧。越靠右越普遍吸引，越靠左越普遍排斥，越靠上越分裂。</p>
        <div class="viz-panel" id="compass"></div>
      </div>
      <div>
        <h2>Domain Prism</h2>
        <p>同一特征在不同领域会改写方向：例如 S3 在学术/金融偏正，但在消费/旅行偏负。</p>
        <div class="viz-panel" id="domain-prism"></div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div>
          <h2>Validity Strips</h2>
          <p>解析率和位置选择率只作为质检背景。Study 1 的核心解释仍以配对 ATE 与 EI 为准。</p>
        </div>
      </div>
      <div class="viz-panel" id="validity"></div>
    </section>

    <section>
      <h2>Paper / PPT Figures</h2>
      <p>这些 PNG 已导出在 <span class="mono">outputs/study1/viz/figures/</span>，可直接放入论文草稿或汇报。</p>
      <div class="figure-grid" id="figures"></div>
    </section>
  </main>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const DATA = __STUDY1_DATA__;
    const fmtPct = (v) => `${(v * 100).toFixed(1)}pp`;
    const fmtRate = (v) => `${(v * 100).toFixed(1)}%`;
    const maxAbsATE = Math.max(...DATA.modelFeature.map(d => Math.abs(d.ATE || 0)), 0.01);
    const maxEI = Math.max(...DATA.modelFeature.map(d => d.EI || 0), 1e-9);
    const tooltip = document.getElementById("tooltip");

    document.getElementById("stat-models").textContent = DATA.callouts.models;
    document.getElementById("stat-features").textContent = DATA.callouts.features;
    document.getElementById("stat-trials").textContent = `${Math.round(DATA.callouts.trials / 1000)}k`;
    document.getElementById("callout-s1").textContent = DATA.callouts.s1Reversal;
    document.getElementById("callout-ei").textContent = DATA.callouts.topLeverage;
    document.getElementById("callout-parse").textContent = DATA.callouts.parseFloor;

    function showTip(event, html) {
      tooltip.innerHTML = html;
      tooltip.style.left = `${event.clientX}px`;
      tooltip.style.top = `${event.clientY}px`;
      tooltip.classList.add("visible");
    }
    function hideTip() { tooltip.classList.remove("visible"); }
    function signedColor(v) {
      const t = Math.max(-1, Math.min(1, v / maxAbsATE));
      const neg = [76, 201, 240], mid = [247, 247, 255], pos = [255, 77, 109];
      const a = t >= 0 ? mid : mid;
      const b = t >= 0 ? pos : neg;
      const w = Math.abs(t);
      const c = a.map((x, i) => Math.round(x * (1 - w) + b[i] * w));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
    function el(name, attrs = {}, children = []) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
      children.forEach(c => node.appendChild(c));
      return node;
    }
    function text(x, y, value, attrs = {}) {
      const t = el("text", {x, y, ...attrs});
      t.textContent = value;
      return t;
    }
    function tipForCell(d) {
      return `<b>${d.model_display}</b><br>${d.factor} ${d.feature_name}<br>ATE: ${fmtPct(d.ATE)} [${fmtPct(d.ci_low)}, ${fmtPct(d.ci_high)}]<br>EI: ${(d.EI || 0).toFixed(4)} / rank ${d.EI_rank || "-"}<br>${d.direction}`;
    }

    function drawConstellation() {
      const w = 720, h = 620, cx = w / 2, cy = h / 2;
      const svg = el("svg", {viewBox: `0 0 ${w} ${h}`, role: "img"});
      const features = DATA.features;
      const order = DATA.modelOrderClustered;
      const angles = new Map(features.map((f, i) => [f.id, -Math.PI / 2 + i * 2 * Math.PI / features.length]));
      order.forEach((m, i) => {
        const r = 136 + i * 18;
        svg.appendChild(el("circle", {cx, cy, r, fill: "none", stroke: "#172842", "stroke-width": 1}));
      });
      features.forEach(f => {
        const a = angles.get(f.id);
        const x = cx + Math.cos(a) * 310;
        const y = cy + Math.sin(a) * 258;
        svg.appendChild(el("line", {x1: cx, y1: cy, x2: x, y2: y, stroke: "#273955", "stroke-width": 1}));
        svg.appendChild(text(x, y, f.id, {"text-anchor": "middle", "dominant-baseline": "middle", class: "feature-label"}));
      });
      DATA.modelFeature.forEach(d => {
        const a = angles.get(d.factor);
        const r = 136 + order.indexOf(d.model_display) * 18;
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r;
        const radius = 4 + 12 * Math.sqrt((d.EI || 0) / maxEI);
        const c = el("circle", {
          cx: x, cy: y, r: radius,
          fill: signedColor(d.ATE || 0),
          stroke: d.significant ? "#f7f7ff" : "#49658d",
          "stroke-width": d.significant ? 2 : 1,
          opacity: 0.95
        });
        c.addEventListener("mousemove", ev => showTip(ev, tipForCell(d)));
        c.addEventListener("mouseleave", hideTip);
        svg.appendChild(c);
      });
      svg.appendChild(text(cx, cy - 8, "Preference", {"text-anchor": "middle", class: "feature-label"}));
      svg.appendChild(text(cx, cy + 12, "Constellation", {"text-anchor": "middle", class: "model-label"}));
      document.getElementById("constellation").replaceChildren(svg);
    }

    let currentOrder = DATA.modelOrderClustered.slice();
    function filteredFeatures() {
      const layer = document.getElementById("feature-filter").value;
      return DATA.features.filter(f => layer === "all" || f.layer === layer);
    }
    function drawHeatmap() {
      const features = filteredFeatures();
      const cellW = 112, cellH = 42, left = 160, top = 42;
      const w = left + features.length * cellW + 20;
      const h = top + currentOrder.length * cellH + 38;
      const svg = el("svg", {viewBox: `0 0 ${w} ${h}`, role: "img"});
      features.forEach((f, x) => {
        svg.appendChild(text(left + x * cellW + cellW / 2, 24, f.id, {"text-anchor": "middle", class: "feature-label"}));
      });
      currentOrder.forEach((m, y) => {
        svg.appendChild(text(8, top + y * cellH + cellH / 2 + 4, m, {class: "model-label"}));
      });
      currentOrder.forEach((m, y) => {
        features.forEach((f, x) => {
          const d = DATA.modelFeature.find(r => r.model_display === m && r.factor === f.id);
          if (!d) return;
          const gx = left + x * cellW, gy = top + y * cellH;
          const rect = el("rect", {x: gx + 2, y: gy + 2, width: cellW - 4, height: cellH - 4, rx: 5, fill: signedColor(d.ATE), opacity: 0.94});
          rect.addEventListener("mousemove", ev => showTip(ev, tipForCell(d)));
          rect.addEventListener("mouseleave", hideTip);
          svg.appendChild(rect);
          const r = 4 + 13 * Math.sqrt((d.EI || 0) / maxEI);
          svg.appendChild(el("circle", {cx: gx + 22, cy: gy + cellH / 2, r, fill: "none", stroke: d.significant ? "#f7f7ff" : "#253d60", "stroke-width": d.significant ? 2 : 1}));
          svg.appendChild(text(gx + cellW - 8, gy + cellH / 2 + 4, (d.ATE >= 0 ? "+" : "") + (d.ATE * 100).toFixed(1), {"text-anchor": "end", fill: "#06101d", "font-size": 11, "font-weight": 700}));
        });
      });
      document.getElementById("heatmap").replaceChildren(svg);
    }

    function drawCompass() {
      const w = 620, h = 390, pad = 54;
      const svg = el("svg", {viewBox: `0 0 ${w} ${h}`, role: "img"});
      const xs = DATA.compass.map(d => d.mean_ATE);
      const ys = DATA.compass.map(d => d.sd_ATE);
      const maxX = Math.max(...xs.map(Math.abs), 0.05);
      const maxY = Math.max(...ys, 0.05);
      const maxMeanEI = Math.max(...DATA.compass.map(d => d.mean_EI), 1e-9);
      const sx = v => pad + (v + maxX) / (2 * maxX) * (w - 2 * pad);
      const sy = v => h - pad - v / maxY * (h - 2 * pad);
      svg.appendChild(el("line", {x1: sx(0), y1: pad, x2: sx(0), y2: h - pad, stroke: "#7d8fb1", "stroke-dasharray": "4 4"}));
      svg.appendChild(el("line", {x1: pad, y1: h - pad, x2: w - pad, y2: h - pad, stroke: "#263b5b"}));
      DATA.compass.forEach(d => {
        const r = 8 + 28 * Math.sqrt(d.mean_EI / maxMeanEI);
        const c = el("circle", {cx: sx(d.mean_ATE), cy: sy(d.sd_ATE), r, fill: d.mean_ATE >= 0 ? "#ffb703" : "#4cc9f0", stroke: "#f7f7ff", "stroke-width": 1.2, opacity: 0.9});
        c.addEventListener("mousemove", ev => showTip(ev, `<b>${d.factor} ${d.feature_name}</b><br>Mean ATE: ${fmtPct(d.mean_ATE)}<br>SD: ${fmtPct(d.sd_ATE)}<br>Mean EI: ${d.mean_EI.toFixed(4)}<br>Positive models: ${d.positive_models} / Negative: ${d.negative_models}`));
        c.addEventListener("mouseleave", hideTip);
        svg.appendChild(c);
        svg.appendChild(text(sx(d.mean_ATE), sy(d.sd_ATE) + 4, d.factor, {"text-anchor": "middle", fill: "#06101d", "font-size": 12, "font-weight": 800}));
      });
      svg.appendChild(text(w / 2, h - 14, "Mean ATE", {"text-anchor": "middle", class: "model-label"}));
      svg.appendChild(text(8, 20, "Disagreement", {class: "model-label"}));
      document.getElementById("compass").replaceChildren(svg);
    }

    function drawDomainPrism() {
      const features = DATA.features;
      const domains = DATA.domains.filter(d => DATA.domainFeature.some(r => r.domain === d.id));
      const cellW = 78, cellH = 44, left = 112, top = 36;
      const w = left + features.length * cellW + 14;
      const h = top + domains.length * cellH + 32;
      const svg = el("svg", {viewBox: `0 0 ${w} ${h}`, role: "img"});
      domains.forEach((d, y) => svg.appendChild(text(8, top + y * cellH + 28, d.label, {class: "model-label"})));
      features.forEach((f, x) => svg.appendChild(text(left + x * cellW + cellW / 2, 22, f.id, {"text-anchor": "middle", class: "feature-label"})));
      domains.forEach((dom, y) => {
        features.forEach((f, x) => {
          const vals = DATA.domainFeature.filter(r => r.domain === dom.id && r.factor === f.id).map(r => r.ATE);
          const mean = vals.reduce((a,b) => a+b, 0) / Math.max(vals.length, 1);
          const rect = el("rect", {x: left + x * cellW + 2, y: top + y * cellH + 2, width: cellW - 4, height: cellH - 4, rx: 5, fill: signedColor(mean), opacity: 0.94});
          rect.addEventListener("mousemove", ev => showTip(ev, `<b>${dom.label} / ${f.id}</b><br>Mean ATE: ${fmtPct(mean)}<br>Models: ${vals.length}`));
          rect.addEventListener("mouseleave", hideTip);
          svg.appendChild(rect);
          svg.appendChild(text(left + x * cellW + cellW / 2, top + y * cellH + 28, (mean * 100).toFixed(1), {"text-anchor": "middle", fill: "#06101d", "font-size": 11, "font-weight": 700}));
        });
      });
      document.getElementById("domain-prism").replaceChildren(svg);
    }

    function drawValidity() {
      const rows = DATA.validity.slice().sort((a, b) => a.model_display.localeCompare(b.model_display));
      const html = `<table><thead><tr><th>Model</th><th>Trials</th><th>Parse OK</th><th>Selection</th><th>Pos 0/1/2</th></tr></thead><tbody>` +
        rows.map(r => {
          const pos = [0,1,2].map(p => r[`position_${p}_selection_rate`]).filter(v => v !== undefined);
          return `<tr><td>${r.model_display}</td><td class="mono">${r.n_trials.toLocaleString()}</td><td><div class="bar-bg"><div class="bar" style="width:${Math.max(0, Math.min(100, r.parse_ok_rate * 100))}%"></div></div><span class="small">${fmtRate(r.parse_ok_rate)}</span></td><td>${fmtRate(r.selection_rate)}</td><td>${pos.map(v => fmtRate(v)).join(" / ")}</td></tr>`;
        }).join("") + `</tbody></table>`;
      document.getElementById("validity").innerHTML = html;
    }

    function drawFigures() {
      const labels = {
        "figures/model_feature_bias_map.png": "Model × Feature Bias Map",
        "figures/structural_hole_constellation.png": "Structural-Hole Constellation",
        "figures/model_fingerprints.png": "Model Fingerprints",
        "figures/leverage_compass.png": "Leverage Compass",
        "figures/domain_prism.png": "Domain Prism",
        "figures/validity_strips.png": "Validity Strips"
      };
      document.getElementById("figures").innerHTML = DATA.figures.map(src => `<figure><img src="${src}" alt="${labels[src] || src}"><figcaption class="small">${labels[src] || src}</figcaption></figure>`).join("");
    }

    document.getElementById("sort-cluster").addEventListener("click", () => {
      currentOrder = DATA.modelOrderClustered.slice();
      document.getElementById("sort-cluster").classList.add("active");
      document.getElementById("sort-original").classList.remove("active");
      drawHeatmap();
    });
    document.getElementById("sort-original").addEventListener("click", () => {
      currentOrder = DATA.modelOrderOriginal.slice();
      document.getElementById("sort-original").classList.add("active");
      document.getElementById("sort-cluster").classList.remove("active");
      drawHeatmap();
    });
    document.getElementById("feature-filter").addEventListener("change", drawHeatmap);

    drawConstellation();
    drawHeatmap();
    drawCompass();
    drawDomainPrism();
    drawValidity();
    drawFigures();
  </script>
</body>
</html>
"""
