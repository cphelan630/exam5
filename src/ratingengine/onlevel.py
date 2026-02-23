"""On-leveling utilities for premium rate level adjustments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _to_rate_change_frame(
    rate_changes: pd.DataFrame | Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    if isinstance(rate_changes, pd.DataFrame):
        frame = rate_changes.copy()
    else:
        frame = pd.DataFrame(rate_changes)

    required = {"effective_date", "change"}
    missing = required - set(frame.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise KeyError(f"rate_changes is missing required columns: {joined}")

    frame = frame.loc[:, ["effective_date", "change"]].copy()
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    frame["change"] = pd.to_numeric(frame["change"], errors="raise")
    if (frame["change"] <= -1.0).any():
        raise ValueError("Each rate change must be greater than -1.0.")
    frame["factor"] = 1.0 + frame["change"]

    grouped = (
        frame.groupby("effective_date", as_index=False)["factor"]
        .prod()
        .sort_values("effective_date")
        .reset_index(drop=True)
    )
    return grouped


def _segment_rate_level(grouped_changes: pd.DataFrame, segment_start: pd.Timestamp) -> float:
    prior = grouped_changes.loc[grouped_changes["effective_date"] <= segment_start, "factor"]
    if prior.empty:
        return 1.0
    return float(prior.prod())


def parallelogram_weights(
    rate_changes: pd.DataFrame | Sequence[Mapping[str, object]],
    eval_period: tuple[object, object],
) -> pd.DataFrame:
    """Compute segment-level parallelogram weights and in-force rate levels."""

    grouped_changes = _to_rate_change_frame(rate_changes)
    eval_start = pd.to_datetime(eval_period[0])
    eval_end = pd.to_datetime(eval_period[1])
    if eval_start >= eval_end:
        raise ValueError("eval_period start must be before eval_period end.")

    total_days = float((eval_end - eval_start) / pd.Timedelta(days=1))
    if total_days <= 0.0:
        raise ValueError("eval_period must have positive length.")

    internal_change_dates = grouped_changes.loc[
        (grouped_changes["effective_date"] > eval_start)
        & (grouped_changes["effective_date"] < eval_end),
        "effective_date",
    ].tolist()
    boundaries = [eval_start, *internal_change_dates, eval_end]

    rows: list[dict[str, object]] = []
    for segment_start, segment_end in zip(boundaries[:-1], boundaries[1:]):
        days = float((segment_end - segment_start) / pd.Timedelta(days=1))
        if days <= 0.0:
            continue
        rows.append(
            {
                "segment_start": segment_start,
                "segment_end": segment_end,
                "days": days,
                "rate_level": _segment_rate_level(grouped_changes, segment_start),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid segments were created for the evaluation period.")
    out["weight"] = out["days"] / total_days
    return out.loc[:, ["segment_start", "segment_end", "days", "weight", "rate_level"]]


def average_rate_level(
    weights: Sequence[float] | pd.Series,
    rate_levels: Sequence[float] | pd.Series,
) -> float:
    """Compute the weighted average rate level."""

    w = np.asarray(weights, dtype=float)
    r = np.asarray(rate_levels, dtype=float)
    if w.shape != r.shape:
        raise ValueError("weights and rate_levels must have matching shapes.")
    if (w < 0.0).any():
        raise ValueError("weights must be non-negative.")
    weight_total = float(w.sum())
    if weight_total <= 0.0:
        raise ValueError("weights must have a positive total.")
    return float(np.dot(w, r) / weight_total)


def onlevel_factor(crl: float, arl: float) -> float:
    """Compute on-level factor from current and average rate levels."""

    current = float(crl)
    average = float(arl)
    if current <= 0.0:
        raise ValueError("crl must be greater than 0.")
    if average <= 0.0:
        raise ValueError("arl must be greater than 0.")
    return current / average
