"""Study 1 publication figures.

Generates three static, publication-quality figures (English labels) from the
per-model study1 outputs and writes them to ``figures/study1/`` as both
300 dpi PNG and vector PDF:

    fig1_leverage_matrix.{png,pdf}   Signed leverage matrix (hero figure)
    fig2_forest.{png,pdf}            Consensus & divergence forest
    fig3_fingerprints.{png,pdf}      Per-model radar fingerprints

Run with the project's conda ``base`` environment:

    python scripts/plot_study1_figures.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Point matplotlib at a writable cache dir before importing pyplot to avoid
# font-cache warnings in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "outputs" / "study1"
FIG_DIR = REPO_ROOT / "figures" / "study1"

FACTOR_ORDER = ["S1", "S2", "S3", "S4", "O1", "O2", "O3", "O4"]
FACTOR_LABELS = {
    "S1": "Evidence\nsolidity",
    "S2": "Dialectical\nframing",
    "S3": "Domain\nexpertise",
    "S4": "Claim\nclarity",
    "O1": "Presentation\nform",
    "O2": "Conclusion-first\nordering",
    "O3": "Structure\nsignposting",
    "O4": "Evidence-claim\nproximity",
}
FACTOR_LABELS_1L = {k: v.replace("\n", " ") for k, v in FACTOR_LABELS.items()}

MODEL_NAMES = {
    "anthropic__claude-haiku-4.5": "Claude-Haiku-4.5",
    "deepseek-v4-flash": "DeepSeek-V4-flash",
    "doubao-seed-2-0-mini-260428": "Doubao-Seed-2.0-mini",
    "google__gemini-3.1-flash-lite": "Gemini-3.1-flash-lite",
    "kimi__kimi-k2.5": "Kimi-K2.5",
    "meta-llama__llama-4-maverick": "Llama-4-Maverick",
    "minimax__MiniMax-M3": "MiniMax-M3",
    "mistralai__mistral-small-2603": "Mistral-Small-2603",
    "openai__gpt-5.4-mini": "GPT-5.4-mini",
    "qwen__qwen3.6-flash": "Qwen3.6-flash",
}


def pretty_model(slug: str) -> str:
    if slug in MODEL_NAMES:
        return MODEL_NAMES[slug]
    return slug.split("__")[-1]


def load_data() -> pd.DataFrame:
    """Return a long dataframe: one row per (model, factor) with ATE, CI and EI_share.

    EI_share is recomputed as EI / sum(EI) within each model so that all models
    (including deepseek-v4-flash, whose ei_leverage.csv predates the EI_share
    column) share one consistent definition.
    """
    ate_frames = []
    for f in sorted(STUDY_DIR.glob("*/ate.csv")):
        slug = f.parent.name
        d = pd.read_csv(f)[["factor", "ATE", "ci_low", "ci_high"]]
        d["model"] = slug
        ate_frames.append(d)
    ate = pd.concat(ate_frames, ignore_index=True)

    ei_frames = []
    for f in sorted(STUDY_DIR.glob("*/ei_leverage.csv")):
        slug = f.parent.name
        d = pd.read_csv(f)[["factor", "EI"]].copy()
        d["EI_share"] = d["EI"] / d["EI"].sum()
        d["model"] = slug
        ei_frames.append(d[["model", "factor", "EI_share"]])
    ei = pd.concat(ei_frames, ignore_index=True)

    df = ate.merge(ei, on=["model", "factor"], how="left")
    df["EI_share"] = df["EI_share"].fillna(0.0)
    df["significant"] = (df["ci_low"] > 0) | (df["ci_high"] < 0)
    df["model_name"] = df["model"].map(pretty_model)
    df["factor"] = pd.Categorical(df["factor"], categories=FACTOR_ORDER, ordered=True)
    return df


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
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / stem}.png / .pdf")


# --------------------------------------------------------------------------- #
# Figure 1: Signed Leverage Matrix (hero)
# --------------------------------------------------------------------------- #
def fig1_leverage_matrix(df: pd.DataFrame):
    ate_p = df.pivot(index="model_name", columns="factor", values="ATE")
    ei_p = df.pivot(index="model_name", columns="factor", values="EI_share")
    sig_p = df.pivot(index="model_name", columns="factor", values="significant")

    # Order columns by mean |ATE| (desc) so content features lead, O-features trail.
    col_order = ate_p.abs().mean(axis=0).sort_values(ascending=False).index.tolist()
    # Order rows by mean |ATE| (desc): most "opinionated" models on top.
    row_order = ate_p.abs().mean(axis=1).sort_values(ascending=True).index.tolist()
    ate_p = ate_p.loc[row_order, col_order]
    ei_p = ei_p.loc[row_order, col_order]
    sig_p = sig_p.loc[row_order, col_order]

    n_rows, n_cols = ate_p.shape
    vmax = float(np.nanmax(np.abs(ate_p.values)))
    cmap = plt.get_cmap("RdBu_r")
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)

    ei_max = float(np.nanmax(ei_p.values))
    smax = 1100.0

    def size(v):
        return 40 + smax * (v / ei_max)

    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    xs, ys, cs, ss, ec, lw = [], [], [], [], [], []
    for i, m in enumerate(ate_p.index):
        for j, c in enumerate(ate_p.columns):
            a = ate_p.loc[m, c]
            xs.append(j)
            ys.append(i)
            cs.append(a)
            ss.append(size(ei_p.loc[m, c]))
            if bool(sig_p.loc[m, c]):
                ec.append("black")
                lw.append(1.3)
            else:
                ec.append("#9a9a9a")
                lw.append(0.5)

    ax.scatter(
        xs, ys, c=cs, s=ss, cmap=cmap, norm=norm,
        edgecolors=ec, linewidths=lw, zorder=3,
    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [f"{c}  {FACTOR_LABELS_1L[c]}" for c in ate_p.columns],
        rotation=32, ha="right", rotation_mode="anchor", fontsize=8.5,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ate_p.index)
    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_axisbelow(True)
    ax.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(
        "Which article features win the bridge position?\n"
        "Signed effect (color) x leverage (size), 10 models x 8 features",
        fontsize=12.5, pad=12,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("ATE  (\u0394 selection rate:  blue = penalized, red = favored)")

    # Size legend (EI share reference bubbles).
    ref = [0.1, 0.3, 0.6]
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=np.sqrt(size(r)),
               markerfacecolor="#bbbbbb", markeredgecolor="#555555",
               label=f"{r:.1f}")
        for r in ref
    ]
    leg1 = ax.legend(
        handles=handles, title="EI share\n(within-model leverage)",
        loc="upper left", bbox_to_anchor=(1.16, 1.0), frameon=False,
        labelspacing=1.4, borderpad=1.0,
    )
    ax.add_artist(leg1)
    sig_handle = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor="#dddddd", markeredgecolor="black",
               markeredgewidth=1.3, label="95% CI excludes 0"),
    ]
    ax.legend(handles=sig_handle, loc="upper left",
              bbox_to_anchor=(1.16, 0.42), frameon=False)

    _save(fig, "fig1_leverage_matrix")


# --------------------------------------------------------------------------- #
# Figure 2: Consensus & Divergence Forest
# --------------------------------------------------------------------------- #
def fig2_forest(df: pd.DataFrame):
    fig, axes = plt.subplots(
        len(FACTOR_ORDER), 1, figsize=(8.2, 10.4), sharex=True
    )
    xmin = float((df["ci_low"].min())) * 100
    xmax = float((df["ci_high"].max())) * 100
    pad = 0.05 * (xmax - xmin)

    pos_c, neg_c = "#b2182b", "#2166ac"
    for ax, fac in zip(axes, FACTOR_ORDER):
        sub = df[df["factor"] == fac].sort_values("ATE").reset_index(drop=True)
        for y, row in sub.iterrows():
            a = row["ATE"] * 100
            lo = row["ci_low"] * 100
            hi = row["ci_high"] * 100
            col = pos_c if a >= 0 else neg_c
            ax.plot([lo, hi], [y, y], color=col, lw=1.4, alpha=0.85, zorder=2)
            if row["significant"]:
                ax.plot(a, y, "o", ms=6.5, color=col,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=3)
            else:
                ax.plot(a, y, "o", ms=6.5, markerfacecolor="white",
                        markeredgecolor=col, markeredgewidth=1.2, zorder=3)
        ax.axvline(0, color="#444444", ls="--", lw=0.9, zorder=1)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["model_name"], fontsize=7.5)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.tick_params(length=0)
        ax.grid(True, axis="x", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
        ax.text(
            0.012, 0.5, f"{fac}  {FACTOR_LABELS_1L[fac]}",
            transform=ax.transAxes, va="center", ha="left",
            fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
        )

    axes[-1].set_xlabel("ATE  (\u0394 selection rate, percentage points)")
    fig.suptitle(
        "Consensus and divergence across models\n"
        "Dialectical framing (S2) is universally penalized; evidence solidity (S1) splits models",
        fontsize=12.5, y=0.995,
    )
    fig.subplots_adjust(top=0.93, hspace=0.28)
    # Shared legend.
    handles = [
        Line2D([0], [0], marker="o", linestyle="", ms=7, color=pos_c,
               markeredgecolor="black", markeredgewidth=0.6, label="Favored, sig."),
        Line2D([0], [0], marker="o", linestyle="", ms=7, color=neg_c,
               markeredgecolor="black", markeredgewidth=0.6, label="Penalized, sig."),
        Line2D([0], [0], marker="o", linestyle="", ms=7, markerfacecolor="white",
               markeredgecolor="#777777", markeredgewidth=1.2, label="CI includes 0"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.012))
    _save(fig, "fig2_forest")


# --------------------------------------------------------------------------- #
# Figure 3: Model fingerprints (radar small multiples)
# --------------------------------------------------------------------------- #
def fig3_fingerprints(df: pd.DataFrame):
    ate_p = df.pivot(index="model_name", columns="factor", values="ATE")[FACTOR_ORDER]
    # Order models by overall attraction (mean ATE) descending for a nice gallery.
    order = ate_p.mean(axis=1).sort_values(ascending=False).index.tolist()
    ate_p = ate_p.loc[order]

    rmax = float(np.nanmax(np.abs(ate_p.values))) * 100
    rmax = np.ceil(rmax / 5) * 5  # round to nearest 5 pp

    n = len(order)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    angles = np.linspace(0, 2 * np.pi, len(FACTOR_ORDER), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    pos_c, neg_c = "#b2182b", "#2166ac"
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 7.2),
        subplot_kw={"projection": "polar"},
    )
    axes = np.atleast_1d(axes).ravel()

    for idx, m in enumerate(order):
        ax = axes[idx]
        vals = ate_p.loc[m].values * 100
        vals_closed = np.concatenate([vals, vals[:1]])
        # Map ATE to radius >= 0, with the baseline circle at ATE = 0.
        r = vals_closed + rmax
        mean_ate = float(np.nanmean(vals))
        line_c = pos_c if mean_ate >= 0 else neg_c
        ax.plot(angles_closed, r, color=line_c, lw=1.6, zorder=3)
        ax.fill(angles_closed, r, color=line_c, alpha=0.22, zorder=2)

        ax.set_ylim(0, 2 * rmax)
        # Baseline (ATE = 0) reference circle.
        ax.plot(np.linspace(0, 2 * np.pi, 200), [rmax] * 200,
                color="#888888", lw=0.8, ls="--", zorder=1)
        ax.set_xticks(angles)
        ax.set_xticklabels(FACTOR_ORDER, fontsize=7.5)
        ax.set_yticks([rmax])
        ax.set_yticklabels([])
        ax.tick_params(pad=-2)
        ax.grid(color="#dddddd", lw=0.5)

        best = ate_p.loc[m].idxmax()
        ax.set_title(
            f"{m}\nstrongest: {best} ({ate_p.loc[m, best]*100:+.0f} pp)",
            fontsize=8.5, pad=18,
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Model fingerprints: per-feature attraction profile (ATE, pp)\n"
        f"Dashed circle = 0; outside = favored, inside = penalized (scale \u00b1{rmax:.0f} pp)",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=4.5, w_pad=1.5)
    _save(fig, "fig3_fingerprints")


# --------------------------------------------------------------------------- #
# Figure 4: Domain x feature annotated heatmap + cross-domain dispersion
# --------------------------------------------------------------------------- #
DOMAIN_ORDER = ["academic_qa", "consumer_product", "finance", "health", "travel"]
DOMAIN_LABELS = {
    "academic_qa": "Academic QA",
    "consumer_product": "Consumer product",
    "finance": "Finance",
    "health": "Health",
    "travel": "Travel",
}


def compute_domain_ate(n_boot: int = 1000) -> pd.DataFrame:
    """Per-(model, domain, factor) paired ATE recomputed from trials.csv.

    Uses the same estimand as study1: parse failures dropped, contrasts confined
    to each factor's own OFAT pairs (target_dim), pair-level differences with a
    query-clustered bootstrap CI. Cached to figures/study1/ate_by_domain.csv and
    reused when newer than every trials.csv.
    """
    cache = FIG_DIR / "ate_by_domain.csv"
    trial_files = sorted(STUDY_DIR.glob("*/trials.csv"))
    if cache.exists():
        cache_mtime = cache.stat().st_mtime
        if all(f.stat().st_mtime < cache_mtime for f in trial_files):
            print(f"using cached {cache}")
            return pd.read_csv(cache)

    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from ai_structural_holes.analysis.ate import paired_ate

    rows = []
    for f in trial_files:
        slug = f.parent.name
        frame = pd.read_csv(f)
        if "parse_ok" in frame.columns:
            frame = frame[frame["parse_ok"] == 1].copy()
        frame["pair_key"] = (
            frame["query_id"].astype(str)
            + "|" + frame["model"].astype(str)
            + "|" + frame["prompt_style"].astype(str)
            + "|" + frame["target_position"].astype(str)
            + "|" + frame["seed"].astype(str)
            + "|" + frame.get("pair_id", "").astype(str)
        )
        for domain in DOMAIN_ORDER:
            d_dom = frame[frame["domain"] == domain]
            for fac in FACTOR_ORDER:
                sub = d_dom[d_dom["target_dim"] == fac]
                if sub.empty:
                    continue
                res = paired_ate(
                    sub, fac, "pair_key", outcome="y",
                    n_boot=n_boot, cluster="query_id",
                )
                rows.append(
                    {
                        "model": slug,
                        "domain": domain,
                        "factor": fac,
                        "ATE": res.ate,
                        "ci_low": res.ci_low,
                        "ci_high": res.ci_high,
                        "n_pairs": sub["pair_key"].nunique(),
                    }
                )
        print(f"  domain ATE done: {slug}")

    out = pd.DataFrame(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"cached -> {cache}")
    return out


def fig4_domain_heatmap(dom: pd.DataFrame, df_overall: pd.DataFrame):
    # Reuse fig1's column ordering (mean |ATE| across models, descending).
    ate_p = df_overall.pivot(index="model_name", columns="factor", values="ATE")
    col_order = ate_p.abs().mean(axis=0).sort_values(ascending=False).index.tolist()

    dom = dom.copy()
    dom["significant"] = (dom["ci_low"] > 0) | (dom["ci_high"] < 0)

    mean_p = dom.pivot_table(index="domain", columns="factor", values="ATE",
                             aggfunc="mean").loc[DOMAIN_ORDER, col_order]

    # Direction consistency: k = models whose ATE shares the cell-mean's sign;
    # s = models individually significant (CI excludes 0).
    k_mat = pd.DataFrame(index=DOMAIN_ORDER, columns=col_order, dtype=float)
    s_mat = pd.DataFrame(index=DOMAIN_ORDER, columns=col_order, dtype=float)
    n_mat = pd.DataFrame(index=DOMAIN_ORDER, columns=col_order, dtype=float)
    for d in DOMAIN_ORDER:
        for fac in col_order:
            cell = dom[(dom["domain"] == d) & (dom["factor"] == fac)]
            mu = mean_p.loc[d, fac]
            k_mat.loc[d, fac] = int((np.sign(cell["ATE"]) == np.sign(mu)).sum())
            s_mat.loc[d, fac] = int(cell["significant"].sum())
            n_mat.loc[d, fac] = len(cell)

    vmax = float(np.nanmax(np.abs(mean_p.values)))
    cmap = plt.get_cmap("RdBu_r")
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(10.5, 7.8), height_ratios=[3.2, 1.0], sharex=False,
        gridspec_kw={"hspace": 0.62},
    )

    # Panel A: annotated heatmap.
    ax.imshow(mean_p.values.astype(float), cmap=cmap, norm=norm, aspect="auto")
    for i, d in enumerate(DOMAIN_ORDER):
        for j, fac in enumerate(col_order):
            mu = mean_p.loc[d, fac] * 100
            k = int(k_mat.loc[d, fac])
            s = int(s_mat.loc[d, fac])
            n = int(n_mat.loc[d, fac])
            dark = abs(mean_p.loc[d, fac]) > 0.55 * vmax
            color = "white" if dark else "black"
            ax.text(j, i - 0.16, f"{mu:+.1f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"{k}/{n} ({s} sig.)", ha="center", va="center",
                    fontsize=7.2, color=color)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(
        [f"{c}  {FACTOR_LABELS_1L[c]}" for c in col_order],
        rotation=32, ha="right", rotation_mode="anchor", fontsize=8.5,
    )
    ax.set_yticks(range(len(DOMAIN_ORDER)))
    ax.set_yticklabels([DOMAIN_LABELS[d] for d in DOMAIN_ORDER], fontsize=9.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(
        "Domain robustness of feature effects\n"
        "Cell: cross-model mean ATE (pp); annotation: models agreeing in sign / total (individually significant)",
        fontsize=12, pad=12,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Mean ATE (\u0394 selection rate)", fontsize=9)

    # Panel B: cross-domain instability per feature.
    rng_vals = (mean_p.max(axis=0) - mean_p.min(axis=0)) * 100  # domain-mean range
    sd_per_model = (
        dom.groupby(["factor", "model"])["ATE"].std().groupby("factor").mean() * 100
    ).reindex(col_order)
    x = np.arange(len(col_order))
    axb.bar(x, rng_vals.values, width=0.55, color="#c7b8e0",
            edgecolor="#6a51a3", linewidth=0.8,
            label="Range of domain means (max\u2212min)")
    axb.plot(x, sd_per_model.values, "D", ms=5.5, color="#6a51a3",
             markeredgecolor="black", markeredgewidth=0.5,
             label="Mean within-model SD across domains", zorder=3)
    worst = rng_vals.idxmax()
    wi = list(col_order).index(worst)
    axb.set_ylim(0, rng_vals.max() * 1.28)
    axb.annotate(
        f"least stable: {worst}",
        xy=(wi, rng_vals[worst]),
        xytext=(wi - 2.2, rng_vals[worst] * 1.08),
        fontsize=8.5, va="center",
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    axb.set_xticks(x)
    axb.set_xticklabels(col_order, fontsize=8.5)
    axb.set_ylabel("pp", fontsize=9)
    axb.set_title("Cross-domain dispersion of the effect", fontsize=10, pad=10)
    axb.legend(fontsize=7.5, frameon=False, loc="upper right")
    axb.grid(True, axis="y", color="#eeeeee", lw=0.6)
    axb.set_axisbelow(True)

    _save(fig, "fig4_domain_heatmap")

    # Console summary: cross-domain sign consistency per factor.
    print("\nCross-domain sign consistency (per factor):")
    for fac in col_order:
        means = mean_p[fac]
        signs = np.sign(means)
        agree = int((signs == signs.mode().iloc[0]).sum())
        print(
            f"  {fac}: domain means {['%+.1f' % (v*100) for v in means.values]} pp"
            f" -> same sign in {agree}/{len(means)} domains"
        )


def main():
    _style()
    df = load_data()
    n_models = df["model"].nunique()
    n_factors = df["factor"].nunique()
    print(f"loaded {n_models} models x {n_factors} factors = {len(df)} cells")
    fig1_leverage_matrix(df)
    fig2_forest(df)
    fig3_fingerprints(df)
    dom = compute_domain_ate()
    fig4_domain_heatmap(dom, df)
    print(f"done -> {FIG_DIR}")


if __name__ == "__main__":
    main()
