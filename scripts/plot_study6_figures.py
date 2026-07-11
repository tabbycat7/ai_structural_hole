"""Study 6 publication figures.

Generates four static, publication-quality figures (English labels) from the
per-model study6 outputs and writes them to ``figures/study6/`` as both
300 dpi PNG and vector PDF:

    fig1_channel_scatter.{png,pdf}   Retrieval-channel ATE vs end-to-end ATE
    fig2_pipeline_dumbbell.{png,pdf} Per-model recall -> cited-rate dumbbells
    fig3_rank_ecdf.{png,pdf}         Target-rank ECDF, treatment vs control
    fig4_study1_vs_study6.{png,pdf}  Slopegraph: choice-only vs end-to-end ATE

Run with the project's conda ``base`` environment:

    python scripts/plot_study6_figures.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
S6_DIR = REPO_ROOT / "outputs" / "study6"
S1_DIR = REPO_ROOT / "outputs" / "study1"
FIG_DIR = REPO_ROOT / "figures" / "study6"

FACTOR_ORDER = ["S1", "S2", "S3", "S4", "O1", "O2", "O3", "O4"]
FACTOR_LABELS = {
    "S1": "Evidence solidity",
    "S2": "Dialectical framing",
    "S3": "Domain expertise",
    "S4": "Claim clarity",
    "O1": "Presentation form",
    "O2": "Conclusion-first ordering",
    "O3": "Structure signposting",
    "O4": "Evidence-claim proximity",
}

# Qualitative palette: S-family warm, O-family cool.
FACTOR_COLORS = {
    "S1": "#b2182b",
    "S2": "#e08214",
    "S3": "#d6604d",
    "S4": "#8c510a",
    "O1": "#2166ac",
    "O2": "#4393c3",
    "O3": "#35978f",
    "O4": "#542788",
}

MODEL_NAMES = {
    # study6 slugs
    "anthropic__claude-haiku-4.5": "Claude-Haiku-4.5",
    "deepseek__deepseek-v4-flash": "DeepSeek-V4-flash",
    "doubao-seed-2-0-mini-260428": "Doubao-Seed-2.0-mini",
    "google__gemini-3.1-flash-lite": "Gemini-3.1-flash-lite",
    "kimi__kimi-k2.5": "Kimi-K2.5",
    "meta-llama__llama-4-maverick": "Llama-4-Maverick",
    "minimax__MiniMax-M3": "MiniMax-M3",
    "mistralai__mistral-small-2603": "Mistral-Small-2603",
    "openai__gpt-5.4-mini": "GPT-5.4-mini",
    "qwen3.6-flash": "Qwen3.6-flash",
    # study1-only slugs (same display names for cross-study alignment)
    "deepseek-v4-flash": "DeepSeek-V4-flash",
    "qwen__qwen3.6-flash": "Qwen3.6-flash",
}


def pretty_model(slug: str) -> str:
    return MODEL_NAMES.get(slug, slug.split("__")[-1])


def _model_dirs():
    return sorted(d for d in S6_DIR.iterdir() if d.is_dir())


def load_ate(name: str) -> pd.DataFrame:
    """Stack per-model <name>.csv (ate_e2e or ate_retrieved) into a long frame."""
    frames = []
    for d in _model_dirs():
        f = d / f"{name}.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)[["factor", "ATE", "ci_low", "ci_high"]]
        t["model_name"] = pretty_model(d.name)
        frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    out["significant"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out


def load_retrieval() -> pd.DataFrame:
    frames = []
    for d in _model_dirs():
        f = d / "retrieval.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)[
            ["query_id", "pair_id", "role", "target_dim", "retrieved", "target_rank"]
        ]
        t["model_name"] = pretty_model(d.name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def load_trials() -> pd.DataFrame:
    frames = []
    for d in _model_dirs():
        f = d / "trials.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)[["query_id", "pair_id", "role", "y", "parse_ok"]]
        t["model_name"] = pretty_model(d.name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def load_study1_ate() -> pd.DataFrame:
    frames = []
    for f in sorted(S1_DIR.glob("*/ate.csv")):
        t = pd.read_csv(f)[["factor", "ATE"]]
        t["model_name"] = pretty_model(f.parent.name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / stem}.png / .pdf")


# --------------------------------------------------------------------------- #
# Figure 1: retrieval-channel ATE vs end-to-end ATE (diagonal scatter)
# --------------------------------------------------------------------------- #
def fig1_channel_scatter(e2e: pd.DataFrame, retr: pd.DataFrame):
    m = e2e.merge(
        retr[["factor", "model_name", "ATE"]],
        on=["factor", "model_name"], suffixes=("_e2e", "_retr"),
    )
    m["x"] = m["ATE_retr"] * 100
    m["y"] = m["ATE_e2e"] * 100

    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    lim_lo = min(m["x"].min(), m["y"].min()) - 2.5
    lim_hi = max(m["x"].max(), m["y"].max()) + 2.5

    # Reference geometry.
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", color="#888888",
            lw=1.0, zorder=1)
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], lim_hi,
                    color="#f7ecec", alpha=0.5, zorder=0)
    ax.fill_between([lim_lo, lim_hi], lim_lo, [lim_lo, lim_hi],
                    color="#ecf1f7", alpha=0.5, zorder=0)
    ax.text(0.02, 0.985, "citation stage amplifies", transform=ax.transAxes,
            fontsize=9, color="#8a3a3a", ha="left", va="top", style="italic")
    ax.text(0.985, 0.02, "citation stage attenuates", transform=ax.transAxes,
            fontsize=9, color="#3a5a8a", ha="right", va="bottom", style="italic")

    for fac in FACTOR_ORDER:
        sub = m[m["factor"] == fac]
        c = FACTOR_COLORS[fac]
        filled = sub[sub["significant"]]
        hollow = sub[~sub["significant"]]
        ax.scatter(filled["x"], filled["y"], s=52, color=c,
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.scatter(hollow["x"], hollow["y"], s=52, facecolors="white",
                   edgecolors=c, linewidths=1.4, zorder=3)
        # Factor centroid marker + label.
        cx, cy = sub["x"].mean(), sub["y"].mean()
        ax.scatter([cx], [cy], marker="X", s=160, color=c,
                   edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate(fac, (cx, cy), xytext=(6, 6), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=c, zorder=5)

    # Notable outlier annotation: GPT-5.4-mini on S1 (hostile in study1 choice,
    # neutral end-to-end).
    out = m[(m["factor"] == "S1") & (m["model_name"] == "GPT-5.4-mini")]
    if len(out):
        ax.annotate(
            "GPT-5.4-mini, S1",
            (out["x"].iloc[0], out["y"].iloc[0]),
            xytext=(10, -22), textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.7),
        )

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Retrieval-channel ATE  (\u0394 P(target in top-k), pp)")
    ax.set_ylabel("End-to-end ATE  (\u0394 P(target cited), pp)")
    ax.set_title(
        "Does the citation stage amplify or erase retrieval advantages?\n"
        "Each point: one model x one feature; X = feature centroid; dashed = y=x",
        pad=12,
    )

    handles = [
        Line2D([0], [0], marker="o", ls="", ms=8, color=FACTOR_COLORS[f],
               markeredgecolor="black", markeredgewidth=0.5,
               label=f"{f}  {FACTOR_LABELS[f]}")
        for f in FACTOR_ORDER
    ] + [
        Line2D([0], [0], marker="o", ls="", ms=8, markerfacecolor="white",
               markeredgecolor="#666666", markeredgewidth=1.4,
               label="e2e CI includes 0"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8.5)
    _save(fig, "fig1_channel_scatter")

    print("\n[fig1] per-factor mean ATE (retrieval -> e2e, pp):")
    for fac in FACTOR_ORDER:
        sub = m[m["factor"] == fac]
        print(f"  {fac}: {sub['x'].mean():+.1f} -> {sub['y'].mean():+.1f}")


# --------------------------------------------------------------------------- #
# Figure 2: pipeline dumbbell (recall -> cited rate per model)
# --------------------------------------------------------------------------- #
def fig2_pipeline_dumbbell(retrieval: pd.DataFrame, trials: pd.DataFrame):
    stats = []
    for mn, r in retrieval.groupby("model_name"):
        t = trials[(trials["model_name"] == mn) & (trials["parse_ok"] == 1)]
        j = t.merge(
            r[["query_id", "pair_id", "role", "retrieved"]],
            on=["query_id", "pair_id", "role"], how="left",
        )
        stats.append(
            {
                "model_name": mn,
                "recall": r["retrieved"].mean(),
                "cited": t["y"].mean(),
                "cite_given_retr": j[j["retrieved"] == 1]["y"].mean(),
            }
        )
    st = pd.DataFrame(stats).sort_values("cited").reset_index(drop=True)

    retr_c, cite_c = "#7f7f7f", "#b2182b"
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for i, row in st.iterrows():
        ax.plot([row["cited"], row["recall"]], [i, i], color="#cccccc",
                lw=2.2, zorder=2)
        ax.plot(row["recall"], i, "o", ms=9, color=retr_c,
                markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        ax.plot(row["cited"], i, "o", ms=9, color=cite_c,
                markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        ax.text(row["recall"] + 0.012, i,
                f"cite|retr = {row['cite_given_retr']:.2f}",
                va="center", fontsize=7.5, color="#555555")

    # Emphasize the narrow retrieval band vs the wide citation band.
    ax.axvspan(st["recall"].min(), st["recall"].max(), color=retr_c,
               alpha=0.10, zorder=0)
    ax.axvspan(st["cited"].min(), st["cited"].max(), color=cite_c,
               alpha=0.07, zorder=0)

    ax.set_yticks(range(len(st)))
    ax.set_yticklabels(st["model_name"])
    ax.set_xlabel("Probability")
    ax.set_xlim(0.5, 0.95)
    ax.grid(True, axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title(
        "The retriever treats all models alike; citation behavior sets them apart\n"
        f"Recall spans {st['recall'].min():.2f}\u2013{st['recall'].max():.2f}; "
        f"end-to-end cited rate spans {st['cited'].min():.2f}\u2013{st['cited'].max():.2f}",
        pad=12,
    )
    handles = [
        Line2D([0], [0], marker="o", ls="", ms=9, color=retr_c,
               markeredgecolor="black", markeredgewidth=0.5,
               label="P(target retrieved into top-k)"),
        Line2D([0], [0], marker="o", ls="", ms=9, color=cite_c,
               markeredgecolor="black", markeredgewidth=0.5,
               label="P(target cited end-to-end)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    _save(fig, "fig2_pipeline_dumbbell")

    print("\n[fig2] recall band: "
          f"{st['recall'].min():.3f}-{st['recall'].max():.3f}; "
          f"cited band: {st['cited'].min():.3f}-{st['cited'].max():.3f}")


# --------------------------------------------------------------------------- #
# Figure 3: target-rank ECDF, treatment vs control (mechanism view)
# --------------------------------------------------------------------------- #
def fig3_rank_ecdf(retrieval: pd.DataFrame):
    top_k = int(retrieval.loc[retrieval["retrieved"] == 1, "target_rank"].max()) + 1
    ranks = np.arange(1, top_k + 1)

    def ecdf(sub: pd.DataFrame) -> np.ndarray:
        # ranks are 0-based; retrieved=0 treated as rank infinity.
        n = len(sub)
        r = sub.loc[sub["retrieved"] == 1, "target_rank"].values
        return np.array([(r <= k - 1).sum() / n for k in ranks])

    t_c, c_c = "#b2182b", "#555555"
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4), sharex=True, sharey=True)
    axes = axes.ravel()
    gains = {}
    for ax, fac in zip(axes, FACTOR_ORDER):
        sub = retrieval[retrieval["target_dim"] == fac]
        et = ecdf(sub[sub["role"] == "treatment"])
        ec = ecdf(sub[sub["role"] == "control"])
        gains[fac] = float(np.mean(et - ec))
        ax.step(ranks, ec, where="post", color=c_c, lw=1.6, label="Control")
        ax.step(ranks, et, where="post", color=t_c, lw=1.6, label="Treatment")
        ax.fill_between(ranks, ec, et, step="post",
                        color=t_c if gains[fac] >= 0 else "#2166ac", alpha=0.18)
        ax.set_title(f"{fac}  {FACTOR_LABELS[fac]}", fontsize=9.5)
        ax.text(0.97, 0.06, f"mean gain {gains[fac]*100:+.1f} pp",
                transform=ax.transAxes, ha="right", fontsize=8,
                color=t_c if gains[fac] >= 0 else "#2166ac")
        ax.set_xticks(ranks)
        ax.grid(True, color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1)

    for i in (4, 5, 6, 7):
        axes[i].set_xlabel("rank r in retrieved top-k")
    for i in (0, 4):
        axes[i].set_ylabel("P(target rank \u2264 r)")
    fig.suptitle(
        "How treatments push the target up the retrieval ranking\n"
        "ECDF of target rank (all 10 models pooled; curve endpoint = recall; "
        "non-retrieved counted as rank \u221e)",
        fontsize=12.5, y=1.0,
    )
    handles = [
        Line2D([0], [0], color=t_c, lw=1.8, label="Treatment (feature at top level)"),
        Line2D([0], [0], color=c_c, lw=1.8, label="Control (baseline level)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.01, 1, 0.92))
    _save(fig, "fig3_rank_ecdf")

    print("\n[fig3] mean ECDF gain per factor (pp): "
          + ", ".join(f"{f}: {g*100:+.1f}" for f, g in gains.items()))


# --------------------------------------------------------------------------- #
# Figure 4: Study 1 (choice-only) vs Study 6 (end-to-end) slopegraph
# --------------------------------------------------------------------------- #
def fig4_study1_vs_study6(e2e: pd.DataFrame, s1: pd.DataFrame):
    m = s1.rename(columns={"ATE": "ate_s1"}).merge(
        e2e.rename(columns={"ATE": "ate_s6"})[["factor", "model_name", "ate_s6"]],
        on=["factor", "model_name"],
    )
    m["ate_s1"] *= 100
    m["ate_s6"] *= 100

    pos_c, neg_c = "#b2182b", "#2166ac"
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6), sharex=True, sharey=True)
    axes = axes.ravel()
    ylim = max(m["ate_s1"].abs().max(), m["ate_s6"].abs().max()) * 1.15

    for ax, fac in zip(axes, FACTOR_ORDER):
        sub = m[m["factor"] == fac]
        for _, row in sub.iterrows():
            col = pos_c if row["ate_s6"] >= 0 else neg_c
            ax.plot([0, 1], [row["ate_s1"], row["ate_s6"]], color=col,
                    lw=0.9, alpha=0.45, zorder=2)
        mu1, mu6 = sub["ate_s1"].mean(), sub["ate_s6"].mean()
        ax.plot([0, 1], [mu1, mu6], color="black", lw=2.6, zorder=3,
                marker="o", ms=6)
        ax.annotate(f"{mu1:+.1f}", (0, mu1), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, fontweight="bold")
        ax.annotate(f"{mu6:+.1f}", (1, mu6), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, fontweight="bold")
        ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Study 1\nchoice-only", "Study 6\nend-to-end"],
                           fontsize=8)
        ax.set_title(f"{fac}  {FACTOR_LABELS[fac]}", fontsize=9.5)
        ax.grid(True, axis="y", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    for i in (0, 4):
        axes[i].set_ylabel("ATE (pp)")
    fig.suptitle(
        "From pure choice to end-to-end RAG: how the pipeline rewrites the story\n"
        "Thin lines: one model each (colored by end-to-end sign); thick line: cross-model mean",
        fontsize=12.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "fig4_study1_vs_study6")

    print("\n[fig4] cross-model mean ATE, study1 -> study6 (pp):")
    for fac in FACTOR_ORDER:
        sub = m[m["factor"] == fac]
        s1r = f"{sub['ate_s1'].min():+.1f}..{sub['ate_s1'].max():+.1f}"
        s6r = f"{sub['ate_s6'].min():+.1f}..{sub['ate_s6'].max():+.1f}"
        print(f"  {fac}: mean {sub['ate_s1'].mean():+.1f} -> "
              f"{sub['ate_s6'].mean():+.1f} | range {s1r} -> {s6r}")


def main():
    _style()
    e2e = load_ate("ate_e2e")
    retr = load_ate("ate_retrieved")
    retrieval = load_retrieval()
    trials = load_trials()
    s1 = load_study1_ate()
    print(
        f"loaded e2e {e2e['model_name'].nunique()} models, "
        f"retrieval rows {len(retrieval)}, trials rows {len(trials)}, "
        f"study1 models {s1['model_name'].nunique()}"
    )
    fig1_channel_scatter(e2e, retr)
    fig2_pipeline_dumbbell(retrieval, trials)
    fig3_rank_ecdf(retrieval)
    fig4_study1_vs_study6(e2e, s1)
    print(f"\ndone -> {FIG_DIR}")


if __name__ == "__main__":
    main()
