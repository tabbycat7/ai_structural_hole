"""Statistical analysis: ATE, mixed-effects logistic regression, summary metrics."""

from .ate import paired_ate, marginal_ate, ate_table
from .regression import mixed_logit, logit_with_clusters
from .metrics import (
    cross_model_consistency,
    position_adjusted_rate,
    deception_gain,
    ei_leverage_table,
    do_route_consistency,
    validity_report,
)
from .heterogeneity import (
    moderation_table,
    interaction_test,
    mediation_proportion,
    position_bias,
)

__all__ = [
    "paired_ate",
    "marginal_ate",
    "ate_table",
    "mixed_logit",
    "logit_with_clusters",
    "cross_model_consistency",
    "position_adjusted_rate",
    "deception_gain",
    "ei_leverage_table",
    "do_route_consistency",
    "validity_report",
    "moderation_table",
    "interaction_test",
    "mediation_proportion",
    "position_bias",
]
