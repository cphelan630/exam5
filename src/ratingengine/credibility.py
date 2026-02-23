"""Credibility helpers for indication blending."""

from __future__ import annotations

import math
from statistics import NormalDist


def limited_fluctuation_z(
    n: float,
    n_full: float | None,
    prob: float = 0.95,
    error: float = 0.05,
) -> float:
    """Compute limited fluctuation credibility weight."""

    n_val = float(n)
    if n_val < 0.0:
        raise ValueError("n must be greater than or equal to 0.")
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must be between 0 and 1.")
    if error <= 0.0:
        raise ValueError("error must be greater than 0.")

    if n_full is None:
        z_crit = NormalDist().inv_cdf((1.0 + float(prob)) / 2.0)
        n_full_val = (z_crit / float(error)) ** 2
    else:
        n_full_val = float(n_full)
        if n_full_val <= 0.0:
            raise ValueError("n_full must be greater than 0 when provided.")

    if n_val == 0.0:
        return 0.0
    return min(1.0, math.sqrt(n_val / n_full_val))


def blend(z: float, indication: float, complement: float) -> float:
    """Blend an indication with a complement using credibility z."""

    z_val = float(z)
    if z_val < 0.0 or z_val > 1.0:
        raise ValueError("z must be between 0 and 1.")
    ind = float(indication)
    comp = float(complement)
    return z_val * ind + (1.0 - z_val) * comp
