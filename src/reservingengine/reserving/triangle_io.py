"""Triangle IO utilities for the Exam 5 reserving workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

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


def project_root() -> Path:
    """Return the repository root based on this module's location."""

    return Path(__file__).resolve().parents[3]


def default_chainladder_data_dir() -> Path:
    """Return the default local chainladder sample-data directory."""

    return project_root() / "chainladder-python" / "chainladder" / "utils" / "data"


def resolve_dataset_path(dataset_name: str, data_dir: Path | None = None) -> Path:
    """Resolve a dataset name to a CSV path in the local chainladder data folder."""

    base = data_dir or default_chainladder_data_dir()
    name = dataset_name if dataset_name.lower().endswith(".csv") else f"{dataset_name}.csv"
    path = Path(base) / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def load_raw_dataset(dataset_name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load a local chainladder sample CSV as a DataFrame."""

    path = resolve_dataset_path(dataset_name, data_dir=data_dir)
    return pd.read_csv(path)


def _triangle_to_frame(triangle: Any, *, origin_as_datetime: bool = False) -> pd.DataFrame:
    """Convert a Triangle to a DataFrame, preserving dimensions when possible."""

    try:
        return triangle.to_frame(origin_as_datetime=origin_as_datetime, keepdims=True)
    except TypeError:
        return triangle.to_frame(origin_as_datetime=origin_as_datetime)


def validate_triangle_totals_match_raw(
    df: pd.DataFrame,
    triangle: Any,
    *,
    raw_value_cols: Sequence[str],
    atol: float = 1e-8,
) -> tuple[bool, float, float]:
    """Validate that raw tabular totals match Triangle totals."""

    raw_total = float(np.nansum(df[list(raw_value_cols)].to_numpy(dtype=float)))
    tri_df = _triangle_to_frame(triangle, origin_as_datetime=False)
    tri_value_cols = [c for c in tri_df.columns if c not in {"origin", "development", "valuation"}]
    tri_total = (
        float(np.nansum(tri_df[tri_value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)))
        if tri_value_cols
        else float("nan")
    )
    return bool(np.isclose(raw_total, tri_total, atol=atol, rtol=0.0)), raw_total, tri_total


def validate_cum_incr_roundtrip(
    triangle: Any,
    *,
    atol: float = 1e-8,
) -> tuple[bool, float]:
    """Validate cum->incr->cum consistency for a Triangle."""

    tri_roundtrip = triangle.cum_to_incr().incr_to_cum()
    original = triangle.to_frame(origin_as_datetime=False).to_numpy(dtype=float)
    reconstructed = tri_roundtrip.to_frame(origin_as_datetime=False).to_numpy(dtype=float)
    diff = np.nanmax(np.abs(original - reconstructed))
    return bool(np.isclose(diff, 0.0, atol=atol, rtol=0.0)), float(diff)


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
