"""Smoke tests: each study runs end-to-end on the mock client."""
import warnings

import pytest

from ai_structural_holes.studies import run_study1, run_study2, run_study3, run_study4

warnings.filterwarnings("ignore")
MODELS = ["mock/a", "mock/b"]


def test_study1_runs():
    r = run_study1(MODELS, per_domain=1, mock=True)
    assert len(r.frame) > 0
    assert {"factor", "EI_norm", "EI_share", "ATE"}.issubset(r.ei.columns)
    assert len(r.ei) == 8  # 8 dimensions


def test_study2_runs():
    r = run_study2(MODELS, per_domain=1, n_points=12, mock=True)
    assert len(r.frame) > 0
    assert "coef" in r.coefficients.columns


def test_study3_runs():
    r = run_study3(MODELS, per_domain=1, prompt_styles=("neutral",), mock=True)
    assert len(r.frame) > 0
    assert "models" in r.consistency


def test_study4_runs():
    r = run_study4(MODELS, per_domain=1, prompt_styles=("neutral",), mock=True)
    assert len(r.frame) > 0
    assert "deception_gain" in r.deception.columns
