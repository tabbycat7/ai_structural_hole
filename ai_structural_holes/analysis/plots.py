"""Plotting helpers (matplotlib). All functions save to a path and return it."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def ei_leverage_bar(ei_table: pd.DataFrame, out_path: Path, title: str = "EI leverage by feature") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ei_table.empty or "EI_share" not in ei_table.columns:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No EI estimates\n(insufficient outcome variation or parse failures)",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    tab = ei_table.sort_values("EI_share", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(tab))))
    ax.barh(tab["factor"], tab["EI_share"])
    ax.set_xlabel("EI share (normalized, sums to 1)")
    ax.set_ylabel("feature")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def ate_forest(ate_table: pd.DataFrame, out_path: Path, title: str = "ATE by feature") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tab = ate_table.dropna(subset=["ATE"]).copy() if "ATE" in ate_table.columns else ate_table.copy()

    if tab.empty:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No ATE estimates\n(insufficient paired outcomes)",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(tab))))
    y = range(len(tab))
    ax.errorbar(
        tab["ATE"], list(y),
        xerr=[tab["ATE"] - tab["ci_low"], tab["ci_high"] - tab["ATE"]],
        fmt="o", capsize=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(tab["factor"])
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("ATE (selection-rate difference)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
