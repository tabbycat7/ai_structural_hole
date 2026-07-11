"""Average Treatment Effect (ATE) estimators.

  - paired_ate: for Study 1's matched counterfactual pairs (same query/base, one
    dimension toggled). Effect = mean(Y_treated) - mean(Y_control), paired by the
    matching key, with a paired bootstrap CI.
  - marginal_ate: difference in selection rate between top vs baseline level of a
    dimension (optionally within strata), with a normal-approx CI.
  - ate_table: ATE for every S/O dimension over a trials frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..codebook import all_ids, get_dimension


@dataclass
class ATE:
    factor: str
    ate: float
    ci_low: float
    ci_high: float
    n_treated: int
    n_control: int
    level_treated: int
    level_control: int

    def as_row(self) -> dict:
        return {
            "factor": self.factor,
            "ATE": self.ate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "level_treated": self.level_treated,
            "level_control": self.level_control,
        }


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                  rng: Optional[np.random.Generator] = None) -> tuple:
    rng = rng or np.random.default_rng(0)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def _cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray,
                          n_boot: int = 2000, alpha: float = 0.05,
                          rng: Optional[np.random.Generator] = None) -> tuple:
    """Cluster (block) bootstrap: resample whole clusters with replacement.

    The per-pair differences within one query are highly correlated, so treating
    them as independent underestimates the CI (pseudo-replication). Here we
    resample the clusters (queries) themselves and pool all differences of each
    drawn cluster, which propagates between-cluster variance correctly.
    """
    rng = rng or np.random.default_rng(0)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    uniq = np.unique(clusters)
    # Pre-group the differences by cluster to avoid repeated masking.
    by_cluster = {c: values[clusters == c] for c in uniq}
    boots = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        pooled = np.concatenate([by_cluster[c] for c in drawn])
        boots.append(pooled.mean())
    boots = np.array(boots)
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def paired_ate(
    df: pd.DataFrame,
    factor: str,
    pair_key: str,
    outcome: str = "y",
    n_boot: int = 2000,
    cluster: Optional[str] = None,
) -> ATE:
    """Paired ATE: within each `pair_key`, treated level vs control level of factor.

    Assumes two levels present for the factor (baseline vs top). Differences are
    computed per pair then averaged. The CI is a paired bootstrap over pairs, or,
    when `cluster` is given (e.g. "query_id"), a cluster bootstrap that resamples
    whole clusters to avoid pseudo-replication.
    """
    dim = get_dimension(factor)
    lo, hi = dim.baseline_code(), dim.top_code()
    sub = df[df[factor].isin([lo, hi])]
    diffs = []
    clusters = []
    have_cluster = bool(cluster) and cluster in sub.columns
    for _, g in sub.groupby(pair_key):
        t = g[g[factor] == hi][outcome]
        c = g[g[factor] == lo][outcome]
        if len(t) and len(c):
            diffs.append(t.mean() - c.mean())
            if have_cluster:
                clusters.append(g[cluster].iloc[0])
    diffs = np.array(diffs, dtype=float)
    ate = float(diffs.mean()) if len(diffs) else float("nan")
    if have_cluster and len(diffs):
        ci = _cluster_bootstrap_ci(diffs, np.array(clusters), n_boot=n_boot)
    else:
        ci = _bootstrap_ci(diffs, n_boot=n_boot)
    n_t = int((sub[factor] == hi).sum())
    n_c = int((sub[factor] == lo).sum())
    return ATE(factor, ate, ci[0], ci[1], n_t, n_c, hi, lo)


def marginal_ate(
    df: pd.DataFrame,
    factor: str,
    outcome: str = "y",
    alpha: float = 0.05,
) -> ATE:
    """Unpaired difference in selection rate: top level vs baseline level."""
    from scipy import stats

    dim = get_dimension(factor)
    lo, hi = dim.baseline_code(), dim.top_code()
    t = df[df[factor] == hi][outcome]
    c = df[df[factor] == lo][outcome]
    if len(t) == 0 or len(c) == 0:
        return ATE(factor, float("nan"), float("nan"), float("nan"), len(t), len(c), hi, lo)
    pt, pc = t.mean(), c.mean()
    ate = float(pt - pc)
    se = np.sqrt(pt * (1 - pt) / len(t) + pc * (1 - pc) / len(c))
    z = stats.norm.ppf(1 - alpha / 2)
    return ATE(factor, ate, ate - z * se, ate + z * se, len(t), len(c), hi, lo)


def ate_table(
    df: pd.DataFrame,
    factors: Optional[Sequence[str]] = None,
    outcome: str = "y",
    paired_key: Optional[str] = None,
    cluster: Optional[str] = None,
) -> pd.DataFrame:
    """ATE for each factor. Uses paired_ate if `paired_key` given, else marginal.

    `cluster` (e.g. "query_id") is forwarded to `paired_ate` for a cluster
    bootstrap CI.
    """
    factors = list(factors or all_ids())
    rows = []
    for f in factors:
        if f not in df.columns:
            continue
        if paired_key and paired_key in df.columns:
            res = paired_ate(df, f, paired_key, outcome, cluster=cluster)
        else:
            res = marginal_ate(df, f, outcome)
        rows.append(res.as_row())
    return pd.DataFrame(rows)
