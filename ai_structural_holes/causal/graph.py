"""Causal graph and confounding sets (plan section 2 / 2.1).

DAG edges (observational generative graph):
    Q -> S, Q -> O, S -> O, Q -> Y, S -> Y, O -> Y, M -> Y, R -> Y

For each treatment X, the confounding (backdoor adjustment) set A_X = variables
that must be adjusted to block all backdoor paths X <- ... -> Y:

    A_S = {Q}              (S <- Q -> Y)
    A_O = {Q, S}           (O <- Q -> Y ; O <- S -> Y)
    A_M = {}               (root)
    A_R = {}               (root)

Sub-dimensions inherit the set of their layer: each S_i adjusts {Q}; each O_j
adjusts {Q, S1..S4}. In a tabular dataset, "adjust for Q" == stratify/condition
on `query_id`; "adjust for S" == condition on the S columns.
"""
from __future__ import annotations

from typing import Dict, List

# Column names used in analysis frames.
Q_COL = "query_id"
S_COLS = ["S1", "S2", "S3", "S4"]
O_COLS = ["O1", "O2", "O3", "O4"]
M_COL = "model"
R_COLS = ["target_position", "set_size", "competitor_quality"]
Y_COL = "y"


EDGES = [
    ("Q", "S"),
    ("Q", "O"),
    ("S", "O"),
    ("Q", "Y"),
    ("S", "Y"),
    ("O", "Y"),
    ("M", "Y"),
    ("R", "Y"),
]


# Confounding set per treatment, expressed as concrete dataframe columns.
CONFOUNDING_SETS: Dict[str, List[str]] = {
    # semantic dims: adjust for Q
    "S1": [Q_COL],
    "S2": [Q_COL],
    "S3": [Q_COL],
    "S4": [Q_COL],
    # structural dims: adjust for Q and S
    "O1": [Q_COL] + S_COLS,
    "O2": [Q_COL] + S_COLS,
    "O3": [Q_COL] + S_COLS,
    "O4": [Q_COL] + S_COLS,
    # roots: no confounders
    M_COL: [],
}


def confounding_set(treatment: str) -> List[str]:
    """Adjustment columns for a treatment. O_j excludes the X itself from S."""
    base = CONFOUNDING_SETS.get(treatment)
    if base is None:
        # default: roots / position factors have empty backdoor set
        return []
    # Never adjust for the treatment itself.
    return [c for c in base if c != treatment]


def describe_graph() -> str:
    lines = ["Edges:"]
    lines += [f"  {a} -> {b}" for a, b in EDGES]
    lines.append("Confounding sets:")
    for x, a in CONFOUNDING_SETS.items():
        lines.append(f"  A_{x} = {{{', '.join(a) if a else '(empty)'}}}")
    return "\n".join(lines)
