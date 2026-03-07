"""Triangle IO utilities for the Exam 5 reserving workflow."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _import_chainladder() -> Any:
    try:
        import chainladder as cl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "chainladder is required for triangle operations. "
            "Install it in your exam5 environment."
        ) from exc
    return cl


def _triangle_to_frame(triangle: Any, *, origin_as_datetime: bool = False) -> pd.DataFrame:
    """Convert a Triangle to a DataFrame, preserving dimensions when possible."""

    try:
        return triangle.to_frame(origin_as_datetime=origin_as_datetime, keepdims=True)
    except TypeError:
        return triangle.to_frame(origin_as_datetime=origin_as_datetime)


def _align_exposure_vector(
    exposure: pd.Series,
    target_origins: Iterable[Any],
) -> np.ndarray:
    """Align an exposure vector to target origin labels."""

    target = pd.Index(list(target_origins))
    aligned = exposure.reindex(target)
    if aligned.isna().any():
        str_exposure = exposure.copy()
        str_exposure.index = str_exposure.index.map(str)
        aligned = str_exposure.reindex(target.map(str))
    if aligned.isna().any():
        missing = target[aligned.isna()].tolist()
        raise ValueError(f"Exposure missing for origins: {missing}")
    return aligned.to_numpy(dtype=float)


def build_exposure_triangle(
    exposure: pd.Series,
    like_triangle: Any,
    *,
    column_name: str = "exposure",
) -> Any:
    """Create a sample-weight Triangle aligned to another Triangle's latest diagonal."""

    sample_weight = like_triangle.latest_diagonal.copy()
    latest = _triangle_to_frame(like_triangle.latest_diagonal, origin_as_datetime=False)
    if "origin" not in latest.columns:
        raise ValueError("Could not locate origin column on latest diagonal frame.")
    aligned_exposure = _align_exposure_vector(exposure.astype(float), latest["origin"])

    values = np.asarray(sample_weight.values, dtype=float).copy()
    if values.shape[-2] != aligned_exposure.shape[0]:
        raise ValueError("Exposure length must match Triangle origin count.")

    for row_idx in range(values.shape[0]):
        values[row_idx, 0, :, 0] = aligned_exposure
    sample_weight.values = values
    sample_weight.columns = [column_name]
    return sample_weight
