"""Mixed-effects / clustered logistic regression for the selection outcome.

Primary model (plan section 7):
    Y ~ S1..S4 + O1..O4 + position + key interactions + (1|query) + (1|model)

statsmodels has no native random-effects *logistic* GLMM that is robust for this;
we provide two practical options:
  - `logit_with_clusters`: plain logistic regression with cluster-robust SEs
    (cluster by query_id), the recommended pragmatic default. Coefficients are
    causal effects under randomization.
  - `mixed_logit`: a Bayesian-free approximation using statsmodels BinomialBayesMixedGLM
    when available, else falls back to clustered logit.
Both return a tidy coefficient table with odds ratios and CIs.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..codebook import all_ids


def _default_formula(
    df: pd.DataFrame,
    outcome: str,
    factors: Sequence[str],
    interactions: Sequence[str],
    include_position: bool,
) -> str:
    terms: List[str] = [f for f in factors if f in df.columns]
    if include_position and "target_position" in df.columns:
        terms.append("target_position")
    terms += list(interactions)
    return f"{outcome} ~ " + " + ".join(terms)


def logit_with_clusters(
    df: pd.DataFrame,
    outcome: str = "y",
    factors: Optional[Sequence[str]] = None,
    interactions: Sequence[str] = (),
    cluster: str = "query_id",
    include_position: bool = True,
) -> pd.DataFrame:
    """Logistic regression with cluster-robust SEs (cluster by `cluster`)."""
    import statsmodels.formula.api as smf

    factors = list(factors or all_ids())
    formula = _default_formula(df, outcome, factors, interactions, include_position)
    work = df.dropna(subset=[outcome]).copy()

    cov_kwds = {}
    fit_kwargs = {"disp": False}
    if cluster in work.columns:
        groups = work[cluster].astype("category").cat.codes.values
        cov_kwds = {"cov_type": "cluster", "cov_kwds": {"groups": groups}}

    model = smf.logit(formula, data=work)
    try:
        res = model.fit(**{**fit_kwargs, **cov_kwds}) if cov_kwds else model.fit(**fit_kwargs)
        tidy = pd.DataFrame(
            {
                "term": res.params.index,
                "coef": res.params.values,
                "se": res.bse.values,
                "z": res.tvalues.values,
                "p": res.pvalues.values,
            }
        )
        tidy["odds_ratio"] = np.exp(tidy["coef"])
        ci = res.conf_int()
        tidy["or_ci_low"] = np.exp(ci[0].values)
        tidy["or_ci_high"] = np.exp(ci[1].values)
        tidy["method"] = "mle"
        return tidy
    except Exception:
        # Perfect separation / non-convergence: fall back to L2-regularized fit.
        res = model.fit_regularized(alpha=1.0, disp=False)
        tidy = pd.DataFrame({"term": res.params.index, "coef": res.params.values})
        tidy["se"] = np.nan
        tidy["z"] = np.nan
        tidy["p"] = np.nan
        tidy["odds_ratio"] = np.exp(tidy["coef"])
        tidy["or_ci_low"] = np.nan
        tidy["or_ci_high"] = np.nan
        tidy["method"] = "l2_regularized(separation)"
        return tidy


def mixed_logit(
    df: pd.DataFrame,
    outcome: str = "y",
    factors: Optional[Sequence[str]] = None,
    interactions: Sequence[str] = (),
    group_vars: Sequence[str] = ("query_id",),
    include_position: bool = True,
) -> pd.DataFrame:
    """Random-intercept logistic GLMM via BinomialBayesMixedGLM when available.

    Falls back to clustered logit on any failure so analysis never blocks.
    """
    factors = list(factors or all_ids())
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        fixed = _default_formula(df, outcome, factors, interactions, include_position)
        vc = {g: f"0 + C({g})" for g in group_vars if g in df.columns}
        work = df.dropna(subset=[outcome]).copy()
        model = BinomialBayesMixedGLM.from_formula(fixed, vc_formulas=vc, data=work)
        res = model.fit_vb()
        tidy = pd.DataFrame(
            {
                "term": res.model.exog_names,
                "coef": np.asarray(res.fe_mean),
                "se": np.asarray(res.fe_sd),
            }
        )
        tidy["odds_ratio"] = np.exp(tidy["coef"])
        tidy["or_ci_low"] = np.exp(tidy["coef"] - 1.96 * tidy["se"])
        tidy["or_ci_high"] = np.exp(tidy["coef"] + 1.96 * tidy["se"])
        return tidy
    except Exception:
        return logit_with_clusters(
            df, outcome, factors, interactions,
            cluster=group_vars[0] if group_vars else "query_id",
            include_position=include_position,
        )
