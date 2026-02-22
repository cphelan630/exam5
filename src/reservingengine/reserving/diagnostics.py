"""Diagnostics for Triangle quality and development assumptions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .triangle_io import triangle_to_frame


def triangle_sanity_checks(triangle: Any) -> pd.DataFrame:
    """Return basic triangle diagnostics for quick quality checks."""

    frame = triangle_to_frame(triangle, origin_as_datetime=False)
    value_cols = [c for c in frame.columns if c not in {"origin", "development", "valuation"}]
    value_col = "values" if "values" in value_cols else value_cols[-1]
    values = pd.to_numeric(frame[value_col], errors="coerce")
    checks = {
        "rows": int(len(frame)),
        "non_null": int(values.notna().sum()),
        "null": int(values.isna().sum()),
        "min": float(np.nanmin(values.to_numpy(dtype=float))),
        "max": float(np.nanmax(values.to_numpy(dtype=float))),
        "total": float(np.nansum(values.to_numpy(dtype=float))),
    }
    return pd.DataFrame([checks])



def selected_ldf_table(triangle: Any, **development_kwargs: Any) -> pd.DataFrame:
    """Fit Development and return selected LDFs as a tidy table."""

    import chainladder as cl

    dev = cl.Development(**development_kwargs).fit(triangle)
    ldf = dev.ldf_.to_frame(origin_as_datetime=False, keepdims=True).reset_index()
    if "values" in ldf.columns:
        ldf = ldf.rename(columns={"values": "selected_ldf"})
    return ldf


def run_correlation_tests(
    triangle: Any,
    *,
    p_critical_development: float = 0.5,
    p_critical_valuation: float = 0.1,
    valuation_total: bool = True,
) -> dict[str, Any]:
    """Run Mack-style development and valuation correlation diagnostics."""

    import chainladder as cl

    dev_corr = cl.DevelopmentCorrelation(triangle, p_critical=p_critical_development)
    val_corr = cl.ValuationCorrelation(
        triangle,
        p_critical=p_critical_valuation,
        total=valuation_total,
    )
    return {
        "development_correlation": dev_corr,
        "valuation_correlation": val_corr,
    }
