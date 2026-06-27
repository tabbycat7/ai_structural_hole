"""Effective Information (EI) attribution (plan section 7, step 2).

Given the unbiased interventional distributions P(Y | do(X=x)) (from backdoor
adjustment or the experimental route), EI quantifies how strongly and specifically
X causally controls Y under a maximum-entropy (uniform) intervention on X:

    EI(X->Y) = (1/|X|) * sum_x  KL( P(Y|do(X=x)) || Pbar(Y) )
    Pbar(Y)  = (1/|X|) * sum_x  P(Y|do(X=x))

This equals the mutual information between a uniform intervention on X and Y, and
decomposes as:

    EI = determinism - degeneracy
    determinism = log2|Y| - <H(Y|do(x))>_x     (higher = more reliable mapping)
    degeneracy  = log2|Y| - H(Pbar(Y))         (higher = causes collapse to same Y)

Normalized effectiveness EI~ = EI / log2|Y| in [0,1] makes factors comparable.
All logs are base-2 (bits).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

_EPS = 1e-12


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        # uniform fallback
        return np.full_like(p, 1.0 / len(p))
    return p / s


def entropy(p: np.ndarray) -> float:
    p = _normalize(p)
    nz = p[p > _EPS]
    return float(-np.sum(nz * np.log2(nz)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize(p)
    q = _normalize(q)
    mask = p > _EPS
    return float(np.sum(p[mask] * np.log2(p[mask] / np.clip(q[mask], _EPS, None))))


@dataclass
class EIResult:
    ei: float
    ei_normalized: float
    determinism: float
    degeneracy: float
    n_states_x: int
    n_states_y: int
    do_table: Dict[object, np.ndarray]

    def as_row(self, name: str = "") -> dict:
        return {
            "factor": name,
            "EI": self.ei,
            "EI_norm": self.ei_normalized,
            "determinism": self.determinism,
            "degeneracy": self.degeneracy,
            "n_states_x": self.n_states_x,
            "n_states_y": self.n_states_y,
        }


def effective_information(do_table: Mapping[object, Sequence[float]]) -> EIResult:
    """Compute EI from a mapping {x: P(Y|do(X=x))}.

    Each value is a probability vector over Y states (e.g. [P(Y=0), P(Y=1)]).
    Uniform intervention over the keys of `do_table` is assumed (max entropy).
    """
    if len(do_table) == 0:
        raise ValueError("empty do_table")
    dists = [(_normalize(np.asarray(v, dtype=float))) for v in do_table.values()]
    lens = {len(d) for d in dists}
    if len(lens) != 1:
        raise ValueError("all effect distributions must have the same length")
    n_y = lens.pop()
    n_x = len(dists)

    p_bar = np.mean(np.vstack(dists), axis=0)  # uniform mixture over interventions

    ei = float(np.mean([kl_divergence(d, p_bar) for d in dists]))

    log_y = np.log2(n_y)
    avg_cond_entropy = float(np.mean([entropy(d) for d in dists]))
    determinism = log_y - avg_cond_entropy
    degeneracy = log_y - entropy(p_bar)
    # numerical: ei == determinism - degeneracy up to float error
    ei_norm = ei / log_y if log_y > 0 else 0.0

    return EIResult(
        ei=ei,
        ei_normalized=ei_norm,
        determinism=determinism,
        degeneracy=degeneracy,
        n_states_x=n_x,
        n_states_y=n_y,
        do_table={k: _normalize(np.asarray(v, dtype=float)) for k, v in do_table.items()},
    )


def ei_from_do_table(do_p1: Mapping[object, float]) -> EIResult:
    """Convenience: EI from {x: P(Y=1|do(X=x))} for binary Y."""
    table = {x: np.array([1.0 - p, p]) for x, p in do_p1.items()}
    return effective_information(table)
