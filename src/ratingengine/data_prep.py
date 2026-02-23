"""Data preparation helpers for ratemaking workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _as_group_list(group_cols: str | Sequence[str] | None) -> list[str]:
    if group_cols is None:
        return []
    if isinstance(group_cols, str):
        return [group_cols]
    return list(group_cols)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"DataFrame is missing required columns: {joined}")


def safe_divide(
    numerator: float | pd.Series | np.ndarray,
    denominator: float | pd.Series | np.ndarray,
) -> float | pd.Series:
    """Divide while returning NaN when denominator is zero."""

    if np.isscalar(numerator) and np.isscalar(denominator):
        den = float(denominator)
        if den == 0.0:
            return float("nan")
        return float(numerator) / den

    num_series = pd.Series(numerator, dtype=float)
    den_series = pd.Series(denominator, dtype=float)
    num_aligned, den_aligned = num_series.align(den_series, join="outer")
    out = num_aligned / den_aligned
    return out.where(den_aligned != 0.0)


def add_time_columns(
    df: pd.DataFrame,
    date_col: str,
    *,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add year/month/quarter columns for a date column."""

    _require_columns(df, [date_col])
    out = df.copy()
    parsed = pd.to_datetime(out[date_col])
    stem = prefix or date_col
    out[f"{stem}_year"] = parsed.dt.year
    out[f"{stem}_month"] = parsed.dt.month
    out[f"{stem}_quarter"] = parsed.dt.quarter
    return out


def aggregate_experience(
    df: pd.DataFrame,
    *,
    group_cols: str | Sequence[str] | None = None,
    exposure_col: str = "exposure",
    premium_col: str = "premium",
    loss_col: str = "loss",
    lae_col: str = "lae",
) -> pd.DataFrame:
    """Aggregate pricing experience and compute core metrics."""

    groups = _as_group_list(group_cols)
    required = [*groups, exposure_col, premium_col, loss_col, lae_col]
    _require_columns(df, required)

    agg_map: dict[str, Any] = {
        exposure_col: "sum",
        premium_col: "sum",
        loss_col: "sum",
        lae_col: "sum",
    }
    if groups:
        grouped = df.groupby(groups, dropna=False, observed=False).agg(agg_map).reset_index()
    else:
        totals = {k: float(df[k].sum()) for k in agg_map}
        grouped = pd.DataFrame([totals])

    grouped["loss_plus_lae"] = grouped[loss_col] + grouped[lae_col]
    grouped["pure_premium"] = safe_divide(grouped["loss_plus_lae"], grouped[exposure_col])
    grouped["loss_ratio"] = safe_divide(grouped["loss_plus_lae"], grouped[premium_col])

    ordered_cols = [
        *groups,
        exposure_col,
        premium_col,
        loss_col,
        lae_col,
        "loss_plus_lae",
        "pure_premium",
        "loss_ratio",
    ]
    return grouped[ordered_cols]
