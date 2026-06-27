"""Sample-size / power analysis for the selection experiments.

Two utilities:
  - `power_two_proportions`: power for detecting a difference between two
    selection rates (paired/independent two-proportion z-test approximation),
    and the inverse `n_for_power`.
  - `n_for_logistic_or`: rough per-condition n to detect a target odds ratio at
    a baseline selection rate (Wald approximation), matching the plan's
    "mixed logistic, OR~=1.5, power=0.8" target.

These are planning approximations, not a substitute for a simulation-based power
analysis on the final mixed-effects model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


def _z(alpha_or_power: float) -> float:
    return stats.norm.ppf(alpha_or_power)


def power_two_proportions(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    """Approximate power of a two-sided two-proportion test with n per group."""
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z_alpha = _z(1 - alpha / 2)
    eff = abs(p1 - p2)
    z_beta = (eff - z_alpha * se0) / se1
    return float(stats.norm.cdf(z_beta))


def n_for_power(p1: float, p2: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Per-group n to reach `power` for a two-proportion test."""
    z_alpha = _z(1 - alpha / 2)
    z_beta = _z(power)
    pbar = (p1 + p2) / 2
    num = (
        z_alpha * math.sqrt(2 * pbar * (1 - pbar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    den = (p1 - p2) ** 2
    return int(math.ceil(num / den))


@dataclass
class LogisticPower:
    baseline_rate: float
    odds_ratio: float
    n_per_condition: int
    power: float


def or_to_p2(p1: float, odds_ratio: float) -> float:
    odds1 = p1 / (1 - p1)
    odds2 = odds1 * odds_ratio
    return odds2 / (1 + odds2)


def n_for_logistic_or(
    baseline_rate: float = 0.2,
    odds_ratio: float = 1.5,
    power: float = 0.8,
    alpha: float = 0.05,
) -> LogisticPower:
    """Per-condition n to detect a target OR at a baseline selection rate."""
    p2 = or_to_p2(baseline_rate, odds_ratio)
    n = n_for_power(baseline_rate, p2, power=power, alpha=alpha)
    achieved = power_two_proportions(baseline_rate, p2, n, alpha=alpha)
    return LogisticPower(baseline_rate, odds_ratio, n, achieved)


if __name__ == "__main__":
    res = n_for_logistic_or(baseline_rate=0.2, odds_ratio=1.5, power=0.8)
    print(
        f"baseline={res.baseline_rate}, OR={res.odds_ratio} -> "
        f"n/condition={res.n_per_condition} (power~{res.power:.2f})"
    )
