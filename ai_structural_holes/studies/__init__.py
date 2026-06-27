"""The four study designs (plan section 4)."""

from .design import (
    ofat_pairs,
    full_factorial,
    fractional_factorial,
    DesignPoint,
)
from .study1_paired import run_study1, Study1Result
from .study2_factorial import run_study2, Study2Result
from .study3_generalization import run_study3, Study3Result
from .study4_adversarial import run_study4, Study4Result

__all__ = [
    "ofat_pairs",
    "full_factorial",
    "fractional_factorial",
    "DesignPoint",
    "run_study1",
    "Study1Result",
    "run_study2",
    "Study2Result",
    "run_study3",
    "Study3Result",
    "run_study4",
    "Study4Result",
]
