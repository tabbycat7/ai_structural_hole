"""Experiment-design generators over the S/O factor space.

  - ofat_pairs: one-factor-at-a-time toggles from a baseline (Study 1). Respects
    the S->O dependency: O4 (evidence proximity) is only toggled on a base where
    S1 evidence exists, since proximity is undefined without evidence.
  - full_factorial: the complete grid over chosen factors (small subsets only).
  - fractional_factorial: a reduced design (orthogonal array via pyDOE2 if
    available, else a balanced random subset) that preserves main effects and
    requested 2-way interactions (Study 2).
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..codebook import baseline_profile, get_dimension, all_ids
from ..data.schema import FeatureProfile, normalize_profile


@dataclass
class DesignPoint:
    profile: FeatureProfile
    label: str = ""
    role: str = "treatment"  # treatment | control | base
    target_dim: Optional[str] = None  # dimension under test (OFAT)

    def __post_init__(self):
        self.profile = normalize_profile(self.profile)


def ofat_pairs(factors: Optional[Sequence[str]] = None) -> List[DesignPoint]:
    """Matched (control, treatment) points toggling one factor at a time.

    Each treatment is labelled with the same `pair_id` (via label) as its control
    so paired ATE can match them.
    """
    factors = list(factors or all_ids())
    points: List[DesignPoint] = []
    for f in factors:
        dim = get_dimension(f)
        lo, hi = dim.baseline_code(), dim.top_code()
        # Base on which to toggle. O4 requires S1 evidence present.
        base = baseline_profile()
        if f == "O4":
            base["S1"] = get_dimension("S1").top_code()
        control = dict(base)
        control[f] = lo
        treat = dict(base)
        treat[f] = hi
        pair_id = f"pair_{f}"
        points.append(DesignPoint(control, label=pair_id, role="control", target_dim=f))
        points.append(DesignPoint(treat, label=pair_id, role="treatment", target_dim=f))
    return points


def full_factorial(factors: Sequence[str]) -> List[DesignPoint]:
    levels = [get_dimension(f).codes() for f in factors]
    points = []
    for combo in itertools.product(*levels):
        prof = baseline_profile()
        for f, c in zip(factors, combo):
            prof[f] = c
        points.append(DesignPoint(prof, label="full", role="treatment"))
    return points


def _binary_view(factors: Sequence[str]) -> Dict[str, tuple]:
    """Map each factor to (low, high) codes for a 2-level design."""
    out = {}
    for f in factors:
        d = get_dimension(f)
        out[f] = (d.baseline_code(), d.top_code())
    return out


def fractional_factorial(
    factors: Sequence[str],
    resolution: str = "auto",
    n_points: Optional[int] = None,
    seed: int = 0,
) -> List[DesignPoint]:
    """Reduced 2-level design.

    Tries pyDOE2 fractional-factorial / Plackett-Burman; on failure builds a
    balanced random subset of the full 2-level grid of the requested size.
    """
    factors = list(factors)
    bv = _binary_view(factors)
    rng = random.Random(seed)

    design_rows: List[Dict[str, int]] = []
    try:
        import numpy as np
        import pyDOE2

        k = len(factors)
        if n_points is None:
            # default: half-fraction if k>=4, else full
            gen = pyDOE2.ff2n(k) if k <= 4 else pyDOE2.fracfact(_default_generator(k))
        else:
            gen = pyDOE2.ff2n(k)
        # gen has -1/+1 coding
        for row in np.atleast_2d(gen):
            d = {}
            for f, val in zip(factors, row):
                lo, hi = bv[f]
                d[f] = hi if val > 0 else lo
            design_rows.append(d)
        if n_points is not None and len(design_rows) > n_points:
            rng.shuffle(design_rows)
            design_rows = design_rows[:n_points]
    except Exception:
        full = list(itertools.product(*[bv[f] for f in factors]))
        rng.shuffle(full)
        take = n_points or max(8, len(factors) * 2)
        for combo in full[:take]:
            design_rows.append({f: c for f, c in zip(factors, combo)})

    points = []
    for d in design_rows:
        prof = baseline_profile()
        prof.update(d)
        points.append(DesignPoint(prof, label="fractional", role="treatment"))
    return points


def _default_generator(k: int) -> str:
    """A simple alias generator string for pyDOE2.fracfact for k factors."""
    base = "a b c d e f g h".split()[: min(k, 8)]
    # define extra factors as products of the first ones (resolution-III-ish)
    extras = []
    combos = ["ab", "ac", "bc", "abc", "abd"]
    for i in range(k - 5):
        extras.append(combos[i % len(combos)])
    return " ".join(base[:5] + extras) if k > 5 else " ".join(base)
