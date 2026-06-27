"""Tests for power analysis utilities."""
from ai_structural_holes.poweranalysis import (
    n_for_logistic_or,
    n_for_power,
    power_two_proportions,
    or_to_p2,
)


def test_or_to_p2_monotone():
    assert or_to_p2(0.2, 1.0) == 0.2
    assert or_to_p2(0.2, 2.0) > 0.2
    assert or_to_p2(0.2, 0.5) < 0.2


def test_n_for_power_achieves_target():
    n = n_for_power(0.2, 0.3, power=0.8)
    achieved = power_two_proportions(0.2, 0.3, n)
    assert achieved >= 0.79


def test_logistic_or_planning():
    res = n_for_logistic_or(baseline_rate=0.2, odds_ratio=1.5, power=0.8)
    assert res.n_per_condition > 0
    assert 0.75 <= res.power <= 0.95
