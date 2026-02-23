"""Loss adjustment utilities for pricing analysis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .types import LossAdjustmentResult


def cap_losses(losses: Iterable[float] | pd.Series, cap: float) -> pd.Series:
    """Cap losses elementwise at a specified threshold."""

    cap_val = float(cap)
    if cap_val < 0.0:
        raise ValueError("cap must be non-negative.")
    series = pd.Series(losses, copy=True, dtype=float)
    return series.clip(upper=cap_val)


def exclude_catastrophes(
    losses: Iterable[float] | pd.Series,
    catastrophe_mask: Iterable[bool] | pd.Series,
) -> pd.Series:
    """Set catastrophe-tagged losses to zero."""

    series = pd.Series(losses, copy=True, dtype=float)
    if isinstance(catastrophe_mask, pd.Series):
        mask = catastrophe_mask.reindex(series.index)
        if mask.isna().any():
            raise ValueError("catastrophe_mask must align cleanly to losses index.")
        mask = mask.astype(bool)
    else:
        mask = pd.Series(list(catastrophe_mask), index=series.index, dtype=bool)

    if len(mask) != len(series):
        raise ValueError("catastrophe_mask length must match losses length.")
    return series.mask(mask, 0.0)


def apply_loss_adjustment_factors(
    loss_total: float,
    *,
    development_factor: float = 1.0,
    trend_factor: float = 1.0,
    benefit_factor: float = 1.0,
    limit_factor: float = 1.0,
    onlevel_factor: float = 1.0,
    cat_load_factor: float = 1.0,
) -> LossAdjustmentResult:
    """Apply multiplicative loss adjustment factors in one chain."""

    components = {
        "development_factor": float(development_factor),
        "trend_factor": float(trend_factor),
        "benefit_factor": float(benefit_factor),
        "limit_factor": float(limit_factor),
        "onlevel_factor": float(onlevel_factor),
        "cat_load_factor": float(cat_load_factor),
    }
    if any(v <= 0.0 for v in components.values()):
        raise ValueError("All multiplicative adjustment factors must be positive.")

    in_total = float(loss_total)
    factor = float(np.prod(list(components.values()), dtype=float))
    adjusted = in_total * factor
    return LossAdjustmentResult(
        input_total=in_total,
        adjusted_total=adjusted,
        adjustment_factor=factor,
        components=components,
    )
