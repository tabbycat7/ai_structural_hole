"""Tests for backdoor adjustment: it should remove confounding bias."""
import numpy as np
import pandas as pd

from ai_structural_holes.causal.backdoor import backdoor_adjust, experimental_do
from ai_structural_holes.causal.graph import confounding_set


def _confounded_data(n=8000, seed=0):
    """Q confounds S->Y. True causal effect of S on Y is fixed; Q biases the
    naive association. Backdoor adjusting for Q should recover the truth.
    """
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 2, n)  # confounder
    # S depends on Q (Q -> S)
    p_s = np.where(q == 1, 0.8, 0.2)
    s = (rng.random(n) < p_s).astype(int)
    # Y depends on both Q and S with KNOWN structure
    logit = -1.0 + 1.0 * s + 2.0 * q
    p_y = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p_y).astype(int)
    return pd.DataFrame({"query_id": q, "S1": s, "y": y})


def test_confounding_set_lookup():
    assert confounding_set("S1") == ["query_id"]
    assert "S1" in confounding_set("O1") and "query_id" in confounding_set("O1")
    assert confounding_set("model") == []


def test_backdoor_recovers_effect_better_than_naive():
    df = _confounded_data()
    naive = experimental_do(df, "S1", "y")  # P(Y|S) ignores Q -> biased
    adjusted = backdoor_adjust(df, "S1", "y", adjustment=["query_id"], method="stratify")

    naive_diff = naive[1] - naive[0]
    adj_diff = adjusted[1] - adjusted[0]
    # ground-truth do-effect computed by g-formula on the known model:
    truth = 0.0
    for qv, pq in [(0, 0.5), (1, 0.5)]:
        for s in (0, 1):
            logit = -1.0 + 1.0 * s + 2.0 * qv
            py = 1 / (1 + np.exp(-logit))
            truth += pq * (py if s == 1 else -py)
    # adjusted estimate should be closer to truth than the naive one
    assert abs(adj_diff - truth) < abs(naive_diff - truth)
    assert abs(adj_diff - truth) < 0.05
