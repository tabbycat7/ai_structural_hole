"""Tests for Effective Information computation."""
import numpy as np

from ai_structural_holes.causal.ei import (
    effective_information,
    ei_from_do_table,
    entropy,
    kl_divergence,
)


def test_deterministic_binary_ei_is_one_bit():
    r = ei_from_do_table({0: 0.0, 1: 1.0})
    assert abs(r.ei - 1.0) < 1e-9
    assert abs(r.ei_normalized - 1.0) < 1e-9


def test_null_cause_has_zero_ei():
    r = ei_from_do_table({0: 0.5, 1: 0.5})
    assert abs(r.ei) < 1e-9


def test_ei_equals_determinism_minus_degeneracy():
    r = ei_from_do_table({0: 0.2, 1: 0.8, 2: 0.5})
    assert abs(r.ei - (r.determinism - r.degeneracy)) < 1e-9


def test_ei_is_nonnegative():
    rng = np.random.default_rng(0)
    for _ in range(50):
        table = {i: float(rng.uniform(0, 1)) for i in range(rng.integers(2, 6))}
        assert ei_from_do_table(table).ei >= -1e-9


def test_entropy_and_kl_basics():
    assert abs(entropy(np.array([0.5, 0.5])) - 1.0) < 1e-9
    assert abs(kl_divergence(np.array([0.5, 0.5]), np.array([0.5, 0.5]))) < 1e-9
    assert kl_divergence(np.array([0.9, 0.1]), np.array([0.5, 0.5])) > 0
