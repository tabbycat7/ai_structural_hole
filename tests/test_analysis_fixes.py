"""Tests: EI table scoping by target_dim and cluster-bootstrap paired ATE."""
import numpy as np
import pandas as pd

from ai_structural_holes.analysis.ate import ate_table, paired_ate
from ai_structural_holes.analysis.metrics import (
    cross_model_consistency,
    ei_leverage_table,
)
from ai_structural_holes.studies.study3_generalization import _ei_by


def _ofat_like_frame() -> pd.DataFrame:
    """S1's own pairs show a positive effect, but S1 is also pinned high inside
    the O4 pairs (with low y), which would drag down an unscoped S1 estimate."""
    rows = []
    # S1 pairs: baseline (0) vs top (2)
    for i in range(100):
        rows.append({"target_dim": "S1", "S1": 0, "O4": 0, "y": int(i < 20)})   # 0.20
        rows.append({"target_dim": "S1", "S1": 2, "O4": 0, "y": int(i < 60)})   # 0.60
    # O4 pairs: S1 held high (2) throughout, y is low regardless of O4
    for i in range(200):
        lvl = 0 if i % 2 == 0 else 2
        rows.append({"target_dim": "O4", "S1": 2, "O4": lvl, "y": int(i < 20)})  # ~0.10
    return pd.DataFrame(rows)


def test_ei_scope_confines_factor_to_its_own_pairs():
    df = _ofat_like_frame()
    scoped = ei_leverage_table(df, factors=["S1"], route="experimental",
                               scope_col="target_dim")
    unscoped = ei_leverage_table(df, factors=["S1"], route="experimental")
    s1_scoped = scoped.set_index("factor").loc["S1", "ATE"]
    s1_unscoped = unscoped.set_index("factor").loc["S1", "ATE"]
    # Scoped S1 effect ~= 0.60 - 0.20 = 0.40; unscoped is diluted by the O4 rows
    assert s1_scoped > s1_unscoped
    assert abs(s1_scoped - 0.40) < 0.05


def test_ei_share_sums_to_one():
    df = _ofat_like_frame()
    tab = ei_leverage_table(df, route="experimental", scope_col="target_dim")
    assert abs(tab["EI_share"].sum() - 1.0) < 1e-9
    assert tab["EI_share"].max() == tab.loc[tab["factor"] == "S1", "EI_share"].iloc[0]


def test_empty_ei_leverage_table_has_schema():
    tab = ei_leverage_table(pd.DataFrame({"y": [0, 0, 0]}), factors=["S1"])
    assert list(tab.columns) == [
        "factor", "EI", "EI_norm", "determinism", "degeneracy",
        "n_states_x", "n_states_y", "ATE", "ate_ci_low", "ate_ci_high",
        "do_p1_min", "do_p1_max", "EI_share",
    ]
    assert tab.empty


def test_study3_ei_by_scopes_to_target_dim():
    """Study 3's per-stratum EI must confine each factor to its own OFAT pairs."""
    df = _ofat_like_frame()
    df["model"] = "m0"
    scoped = _ei_by(df, "model", scope_col="target_dim")
    unscoped = _ei_by(df, "model", scope_col=None)
    s1_scoped = scoped.set_index("factor").loc["S1", "ATE"]
    s1_unscoped = unscoped.set_index("factor").loc["S1", "ATE"]
    assert abs(s1_scoped - 0.40) < 0.05
    assert s1_scoped > s1_unscoped


def test_cross_model_consistency_forwards_scope_col():
    """cross_model_consistency must apply target_dim scoping when requested."""
    df = _ofat_like_frame()
    # Two models with identical data so rankings are trivially concordant; the
    # point is that scoped EI values feed the ranking, not the contaminated ones.
    df["model"] = "m0"
    df2 = df.copy()
    df2["model"] = "m1"
    both = pd.concat([df, df2], ignore_index=True)
    res = cross_model_consistency(
        both, factors=["S1", "O4"], route="experimental", scope_col="target_dim"
    )
    ei_var = res["ei_variance"].set_index("factor")
    # Scoped S1 EI reflects the 0.20->0.60 contrast, not the diluted whole-frame one.
    scoped_s1 = ei_leverage_table(
        df, factors=["S1"], route="experimental", scope_col="target_dim"
    ).set_index("factor").loc["S1", "EI_norm"]
    assert abs(ei_var.loc["S1", "EI_norm_mean"] - scoped_s1) < 1e-9


def _paired_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for q in range(8):
        for p in range(4):
            pk = f"q{q}|p{p}"
            rows.append({"query_id": f"q{q}", "pair_key": pk, "S2": 0,
                         "y": int(rng.random() < 0.3)})
            rows.append({"query_id": f"q{q}", "pair_key": pk, "S2": 1,
                         "y": int(rng.random() < 0.6)})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_ci_runs_and_brackets_estimate():
    df = _paired_frame()
    res = paired_ate(df, "S2", pair_key="pair_key", cluster="query_id")
    assert np.isfinite(res.ate)
    assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)
    assert res.ci_low <= res.ate <= res.ci_high


def test_ate_table_forwards_cluster():
    df = _paired_frame()
    tab = ate_table(df, factors=["S2"], paired_key="pair_key", cluster="query_id")
    assert "ATE" in tab.columns
    row = tab.set_index("factor").loc["S2"]
    assert row["ci_low"] <= row["ATE"] <= row["ci_high"]
