"""Heterogeneity, moderation, mediation, and position-bias analyses (plan 7).

  - moderation_table: per-stratum ATE/EI to test S x M, S x domain, S x prompt.
  - interaction_test: coefficient of an X:Z interaction term in a clustered logit.
  - mediation_proportion: does O4 (evidence-claim proximity) amplify S1's effect?
    Estimates the share of S1's total effect that flows through / is moderated by
    O4 by comparing the S1 effect with vs without O4 present (moderation framing).
  - position_bias: selection rate by target position + a logit slope on position.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..codebook import get_dimension
from .ate import marginal_ate
from .metrics import ei_leverage_table


def moderation_table(
    df: pd.DataFrame,
    factor: str,
    moderator: str,
    outcome: str = "y",
) -> pd.DataFrame:
    """ATE of `factor` within each level of `moderator` (e.g. model / domain)."""
    rows = []
    for level, g in df.groupby(moderator):
        res = marginal_ate(g, factor, outcome)
        row = res.as_row()
        row[moderator] = level
        rows.append(row)
    return pd.DataFrame(rows)


def interaction_test(
    df: pd.DataFrame,
    factor: str,
    moderator: str,
    outcome: str = "y",
    cluster: str = "query_id",
) -> pd.DataFrame:
    """Clustered logit with an explicit factor:moderator interaction term."""
    from .regression import logit_with_clusters

    moderator_term = moderator
    # treat non-numeric moderators as categorical
    if df[moderator].dtype == object:
        moderator_term = f"C({moderator})"
    interaction = f"{factor}:{moderator_term}"
    tidy = logit_with_clusters(
        df, outcome=outcome, factors=[factor, moderator_term],
        interactions=[interaction], cluster=cluster, include_position=False,
    )
    tidy["model_spec"] = f"{outcome} ~ {factor} * {moderator_term}"
    return tidy


def mediation_proportion(
    df: pd.DataFrame,
    cause: str = "S1",
    moderator: str = "O4",
    outcome: str = "y",
) -> Dict[str, float]:
    """Moderation framing: S1 effect with O4 present vs absent.

    Returns the S1 ATE in the O4-present subset and O4-absent subset, plus the
    amplification (difference). A large positive amplification supports the
    hypothesis that proximity (O4) strengthens evidence (S1) influence.
    """
    o4 = get_dimension(moderator)
    present_codes = [c for c in o4.codes() if c > o4.baseline_code()]
    sub_present = df[df[moderator].isin(present_codes)]
    sub_absent = df[df[moderator] == o4.baseline_code()]

    ate_present = marginal_ate(sub_present, cause, outcome).ate if len(sub_present) else float("nan")
    ate_absent = marginal_ate(sub_absent, cause, outcome).ate if len(sub_absent) else float("nan")
    amplification = (
        ate_present - ate_absent
        if not (np.isnan(ate_present) or np.isnan(ate_absent))
        else float("nan")
    )
    return {
        "cause": cause,
        "moderator": moderator,
        "ate_moderator_present": ate_present,
        "ate_moderator_absent": ate_absent,
        "amplification": amplification,
    }


def position_bias(df: pd.DataFrame, outcome: str = "y") -> Dict[str, object]:
    """Selection rate by target position + a logistic slope on position."""
    by_pos = (
        df.groupby("target_position")[outcome].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "selection_rate", "count": "n"})
    )
    slope = float("nan")
    try:
        import statsmodels.formula.api as smf

        res = smf.logit(f"{outcome} ~ target_position", data=df).fit(disp=False)
        slope = float(res.params.get("target_position", float("nan")))
    except Exception:
        pass
    return {"by_position": by_pos, "logit_slope_position": slope}
