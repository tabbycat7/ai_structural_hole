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
from .study4_adversarial import run_study4, Study4Result, build_study4_materials
from .study5_rag import run_study5, Study5Result
from .study6_query_rewrite import run_study6, Study6Result, retrieval_by_model

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
    "build_study4_materials",
    "run_study5",
    "Study5Result",
    "run_study6",
    "Study6Result",
    "retrieval_by_model",
]
