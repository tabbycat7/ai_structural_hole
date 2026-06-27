"""Summary evaluation metrics (plan section 8).

  - ei_leverage_table: the headline output. For each factor, run backdoor
    adjustment (or experimental route) -> P(Y|do(X)) -> EI, and assemble a ranked
    "structural-hole leverage" table (EI, normalized EI, determinism, degeneracy,
    plus the ATE for direction/magnitude).
  - cross_model_consistency: variance / Kendall agreement of factor rankings
    across models.
  - position_adjusted_rate: selection rate averaged over target positions.
  - deception_gain / vulnerability: Study 4 reverse-direction metrics.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..codebook import all_ids
from ..causal.backdoor import backdoor_adjust
from ..causal.ei import ei_from_do_table
from .ate import marginal_ate


def ei_leverage_table(
    df: pd.DataFrame,
    factors: Optional[Sequence[str]] = None,
    outcome: str = "y",
    method: str = "gcomp",
    route: str = "backdoor",
) -> pd.DataFrame:
    """Per-factor EI leverage table, sorted by normalized EI descending.

    route: "backdoor" (adjust for the factor's confounding set) or
           "experimental" (P(Y|do(X)) = P(Y|X), for randomized data).
    """
    factors = list(factors or all_ids())
    rows = []
    for f in factors:
        if f not in df.columns or df[f].nunique() < 2:
            continue
        if route == "experimental":
            do_p = backdoor_adjust(df, f, outcome, adjustment=[], method=method)
        else:
            do_p = backdoor_adjust(df, f, outcome, method=method)
        do_p = {k: v for k, v in do_p.items() if not np.isnan(v)}
        if len(do_p) < 2:
            continue
        ei = ei_from_do_table(do_p)
        ate = marginal_ate(df, f, outcome)
        row = ei.as_row(name=f)
        row["ATE"] = ate.ate
        row["ate_ci_low"] = ate.ci_low
        row["ate_ci_high"] = ate.ci_high
        row["do_p1_min"] = float(min(do_p.values()))
        row["do_p1_max"] = float(max(do_p.values()))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("EI_norm", ascending=False).reset_index(drop=True)
    return out


def position_adjusted_rate(df: pd.DataFrame, by: Sequence[str] = (), outcome: str = "y") -> pd.DataFrame:
    """Selection rate averaged uniformly over target positions (controls R)."""
    by = list(by)
    group_cols = by + ["target_position"]
    per_pos = df.groupby(group_cols)[outcome].mean().reset_index()
    if by:
        return per_pos.groupby(by)[outcome].mean().reset_index(name="adj_rate")
    return pd.DataFrame({"adj_rate": [per_pos[outcome].mean()]})


def cross_model_consistency(
    df: pd.DataFrame,
    factors: Optional[Sequence[str]] = None,
    outcome: str = "y",
    model_col: str = "model",
) -> Dict[str, object]:
    """Agreement of per-factor EI rankings across models (Kendall's W + variance)."""
    from scipy import stats

    factors = list(factors or all_ids())
    rank_by_model = {}
    ei_by_model = {}
    for m, g in df.groupby(model_col):
        tab = ei_leverage_table(g, factors, outcome)
        if tab.empty:
            continue
        tab = tab.set_index("factor")
        ei_by_model[m] = tab["EI_norm"].to_dict()
        order = tab.sort_values("EI_norm", ascending=False).index.tolist()
        rank_by_model[m] = {f: r for r, f in enumerate(order)}

    common = sorted(set.intersection(*[set(d) for d in rank_by_model.values()])) if rank_by_model else []
    result: Dict[str, object] = {"models": list(rank_by_model.keys()), "common_factors": common}

    if len(rank_by_model) >= 2 and len(common) >= 2:
        rank_matrix = np.array([[rank_by_model[m][f] for f in common] for m in rank_by_model])
        # Kendall's W (coefficient of concordance)
        n = rank_matrix.shape[1]
        k = rank_matrix.shape[0]
        col_sums = rank_matrix.sum(axis=0)
        S = np.sum((col_sums - col_sums.mean()) ** 2)
        W = 12 * S / (k ** 2 * (n ** 3 - n)) if (n ** 3 - n) > 0 else float("nan")
        result["kendall_w"] = float(W)
        # mean pairwise Spearman
        corrs = []
        models = list(rank_by_model.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a = [rank_by_model[models[i]][f] for f in common]
                b = [rank_by_model[models[j]][f] for f in common]
                corrs.append(stats.spearmanr(a, b).correlation)
        result["mean_spearman"] = float(np.nanmean(corrs)) if corrs else float("nan")
    # EI variance per factor across models
    if ei_by_model:
        var_rows = []
        for f in common:
            vals = [ei_by_model[m][f] for m in ei_by_model if f in ei_by_model[m]]
            var_rows.append({"factor": f, "EI_norm_mean": np.mean(vals), "EI_norm_var": np.var(vals)})
        result["ei_variance"] = pd.DataFrame(var_rows)
    return result


def do_route_consistency(
    df: pd.DataFrame,
    factors: Optional[Sequence[str]] = None,
    outcome: str = "y",
) -> pd.DataFrame:
    """Compare EI from experimental-do vs backdoor-adjusted routes per factor.

    Large discrepancies flag residual confounding or insufficient overlap; small
    ones support cross-validation of the two identification routes (plan 7).
    """
    factors = list(factors or all_ids())
    exp = ei_leverage_table(df, factors, outcome, route="experimental").set_index("factor")["EI_norm"]
    bd = ei_leverage_table(df, factors, outcome, route="backdoor").set_index("factor")["EI_norm"]
    common = sorted(set(exp.index) & set(bd.index))
    rows = []
    for f in common:
        rows.append({
            "factor": f,
            "EI_experimental": float(exp[f]),
            "EI_backdoor": float(bd[f]),
            "abs_diff": abs(float(exp[f]) - float(bd[f])),
        })
    return pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)


def validity_report(df: pd.DataFrame) -> Dict[str, object]:
    """Operational validity stats: parse rate, position spread, set-size coverage."""
    rep: Dict[str, object] = {
        "n_trials": int(len(df)),
        "parse_ok_rate": float(df["parse_ok"].mean()) if "parse_ok" in df else float("nan"),
        "overall_selection_rate": float(df["y"].mean()) if "y" in df else float("nan"),
    }
    if "target_position" in df:
        rep["positions_covered"] = sorted(df["target_position"].unique().tolist())
    if "model" in df:
        rep["n_models"] = int(df["model"].nunique())
    if "domain" in df:
        rep["domains"] = sorted(df["domain"].dropna().unique().tolist())
    return rep


def deception_gain(
    df: pd.DataFrame,
    factor: str,
    outcome: str = "y",
    authenticity_col: str = "authenticity",
) -> Dict[str, float]:
    """Study 4: how much faking `factor` raises selection vs genuine/absent.

    Compares selection rate of:
      - none:    factor at baseline (control)
      - genuine: factor present and verifiable
      - fake:    factor present but fabricated
    deception_gain = P(sel|fake) - P(sel|none)
    discount      = P(sel|genuine) - P(sel|fake)  (model's penalty on fakery)
    vulnerability = P(sel|fake) / P(sel|genuine)  (>~1 == easily fooled)
    """
    from ..codebook import get_dimension

    dim = get_dimension(factor)
    base, top = dim.baseline_code(), dim.top_code()
    none = df[df[factor] == base][outcome]
    present = df[df[factor] == top]
    genuine = present[present[authenticity_col] == "genuine"][outcome]
    fake = present[present[authenticity_col] == "fake"][outcome]

    def _m(s):
        return float(s.mean()) if len(s) else float("nan")

    p_none, p_gen, p_fake = _m(none), _m(genuine), _m(fake)
    out = {
        "factor": factor,
        "p_none": p_none,
        "p_genuine": p_gen,
        "p_fake": p_fake,
        "deception_gain": p_fake - p_none,
        "discount": p_gen - p_fake,
        "vulnerability": (p_fake / p_gen) if p_gen and not np.isnan(p_gen) else float("nan"),
    }
    return out
