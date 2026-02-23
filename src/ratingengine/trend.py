"""Trending utilities for ratemaking exhibits."""

from __future__ import annotations

import pandas as pd

from .types import TrendResult


def _validate_annual_rate(rate: float, *, name: str) -> float:
    val = float(rate)
    if val <= -1.0:
        raise ValueError(f"{name} must be greater than -1.0.")
    return val


def _year_fraction(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return float((end - start) / pd.Timedelta(days=365.25))


def trend_to_average_date(
    value: float,
    from_date: object,
    to_date: object,
    annual_rate: float,
) -> TrendResult:
    """Trend a value from one average date to another."""

    base_level = float(value)
    start = pd.to_datetime(from_date)
    end = pd.to_datetime(to_date)
    rate = _validate_annual_rate(annual_rate, name="annual_rate")

    years = _year_fraction(start, end)
    factor = float((1.0 + rate) ** years)
    return TrendResult(
        base_level=base_level,
        years_retro=years,
        years_pros=0.0,
        retro_factor=factor,
        pros_factor=1.0,
        total_factor=factor,
        trended_level=base_level * factor,
    )


def two_step_trend(
    base_level: float,
    mid_period_date: object,
    target_date: object,
    retro_trend: float,
    pros_trend: float,
    *,
    as_of_date: object,
) -> TrendResult:
    """Apply retrospective then prospective trend using an explicit split date."""

    level = float(base_level)
    mid = pd.to_datetime(mid_period_date)
    as_of = pd.to_datetime(as_of_date)
    target = pd.to_datetime(target_date)
    retro_rate = _validate_annual_rate(retro_trend, name="retro_trend")
    pros_rate = _validate_annual_rate(pros_trend, name="pros_trend")

    if mid > as_of:
        raise ValueError("mid_period_date must be on or before as_of_date.")
    if as_of > target:
        raise ValueError("as_of_date must be on or before target_date.")

    years_retro = _year_fraction(mid, as_of)
    years_pros = _year_fraction(as_of, target)
    retro_factor = float((1.0 + retro_rate) ** years_retro)
    pros_factor = float((1.0 + pros_rate) ** years_pros)
    total_factor = retro_factor * pros_factor
    return TrendResult(
        base_level=level,
        years_retro=years_retro,
        years_pros=years_pros,
        retro_factor=retro_factor,
        pros_factor=pros_factor,
        total_factor=total_factor,
        trended_level=level * total_factor,
    )
