"""do-calculus backdoor adjustment (Theorem 1) and the experimental do route.

Theorem 1:  P(Y | do(X)) = sum_{A_X} P(Y | X, A_X) * P(A_X)

Two estimators of P(Y=1 | do(X=x)):
  - "stratify": literal stratified adjustment. Averages P(Y|X,A) over the
    empirical distribution of the adjustment set A. Exact but data-hungry when A
    has many cells (e.g. many query ids).
  - "gcomp": g-computation / standardization via logistic regression
    Y ~ X + A, then average predicted P(Y=1|X=x, A=a_i) over observed rows.
    Far more data-efficient and is the recommended default for high-cardinality A.

`experimental_do` is the randomized route: under do() randomization the backdoor
set is empty so P(Y|do(X)) = P(Y|X) (plain conditional means).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .graph import confounding_set


def _states(series: pd.Series) -> List:
    return sorted(series.dropna().unique().tolist())


def experimental_do(df: pd.DataFrame, treatment: str, outcome: str = "y") -> Dict:
    """P(Y=1|do(X=x)) = P(Y=1|X=x) under randomized intervention (empty backdoor)."""
    out = {}
    for x, g in df.groupby(treatment):
        out[x] = float(g[outcome].mean())
    return out


def _stratify_adjust(
    df: pd.DataFrame, treatment: str, adjustment: Sequence[str], outcome: str
) -> Dict:
    states = _states(df[treatment])
    adj = list(adjustment)
    if not adj:
        return experimental_do(df, treatment, outcome)

    # P(A) over observed adjustment-set cells.
    cell_counts = df.groupby(adj).size()
    total = cell_counts.sum()
    pA = (cell_counts / total).to_dict()

    do_p = {}
    for x in states:
        sub = df[df[treatment] == x]
        if sub.empty:
            do_p[x] = float("nan")
            continue
        cond = sub.groupby(adj)[outcome].mean()  # P(Y|X=x, A)
        acc = 0.0
        wsum = 0.0
        for cell, p_y in cond.items():
            w = pA.get(cell, 0.0)
            if np.isnan(p_y):
                continue
            acc += w * p_y
            wsum += w
        # renormalize over cells actually observed for this x (defensive)
        do_p[x] = float(acc / wsum) if wsum > 0 else float("nan")
    return do_p


def _gcomp_adjust(
    df: pd.DataFrame, treatment: str, adjustment: Sequence[str], outcome: str
) -> Dict:
    import statsmodels.formula.api as smf

    adj = list(adjustment)
    states = _states(df[treatment])
    work = df[[treatment, outcome] + adj].dropna().copy()

    # Build formula with categorical handling for non-numeric adjustment cols.
    rhs_terms = [f"C({treatment})"]
    for a in adj:
        if work[a].dtype == object or str(work[a].dtype).startswith("category"):
            rhs_terms.append(f"C({a})")
        else:
            rhs_terms.append(a)
    formula = f"{outcome} ~ " + " + ".join(rhs_terms)

    model = smf.logit(formula, data=work).fit(disp=False)

    do_p = {}
    for x in states:
        cf = work.copy()
        cf[treatment] = x
        preds = model.predict(cf)
        do_p[x] = float(preds.mean())
    return do_p


def backdoor_adjust(
    df: pd.DataFrame,
    treatment: str,
    outcome: str = "y",
    adjustment: Optional[Sequence[str]] = None,
    method: str = "gcomp",
) -> Dict:
    """Return {x: P(Y=1 | do(treatment=x))} via backdoor adjustment.

    adjustment defaults to the project confounding set for `treatment`.
    method: "gcomp" (default, model-based) or "stratify" (literal Theorem 1).
    """
    if adjustment is None:
        adjustment = confounding_set(treatment)
    adjustment = [a for a in adjustment if a in df.columns and a != treatment]

    if not adjustment:
        return experimental_do(df, treatment, outcome)
    if method == "stratify":
        return _stratify_adjust(df, treatment, adjustment, outcome)
    if method == "gcomp":
        try:
            return _gcomp_adjust(df, treatment, adjustment, outcome)
        except Exception:
            # fall back to stratification if the model fails to fit
            return _stratify_adjust(df, treatment, adjustment, outcome)
    raise ValueError(f"unknown method: {method}")


def do_distribution(
    df: pd.DataFrame,
    treatment: str,
    outcome: str = "y",
    adjustment: Optional[Sequence[str]] = None,
    method: str = "gcomp",
) -> Dict[object, np.ndarray]:
    """Full effect distribution P(Y|do(X=x)) as a 2-vector [P(Y=0), P(Y=1)] per x.

    Suitable as direct input to the EI computation.
    """
    p1 = backdoor_adjust(df, treatment, outcome, adjustment, method)
    return {x: np.array([1.0 - p, p]) for x, p in p1.items() if not np.isnan(p)}
