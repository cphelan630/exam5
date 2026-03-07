"""Reserving methods split into Track A and Track B workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .triangle_io import _import_chainladder, _triangle_to_frame, build_exposure_triangle

_META_COLUMNS = {"origin", "development", "valuation"}


@dataclass
class MethodResult:
    """Compact summary for one reserving method run."""

    method_name: str
    ultimate_total: float
    ibnr_total: float
    latest_reported_total: float
    assumption_notes: str


def _detect_value_column(frame: pd.DataFrame) -> str:
    candidates = [c for c in frame.columns if c not in _META_COLUMNS]
    if "values" in candidates:
        return "values"
    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(frame[c])]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    if len(numeric_candidates) > 1:
        return numeric_candidates[-1]
    if candidates:
        return candidates[-1]
    raise ValueError("Could not detect Triangle value column.")


def _segment_key_from_row(row: pd.Series, segment_cols: Sequence[str]) -> str:
    if not segment_cols:
        return "(All)"
    if segment_cols == ["Total"] and str(row.get("Total")) == "Total":
        return "(All)"
    parts = [f"{col}={row[col]}" for col in segment_cols]
    return "|".join(parts)


def _triangle_series_by_origin(triangle: Any) -> pd.Series:
    frame = _triangle_to_frame(triangle, origin_as_datetime=False)
    segment_cols = [name for name in frame.index.names if name is not None]
    reset = frame.reset_index()
    reset["segment_key"] = reset.apply(
        lambda row: _segment_key_from_row(row, segment_cols), axis=1
    )
    if "origin" not in reset.columns:
        raise ValueError("Triangle frame is missing required origin column.")
    value_col = _detect_value_column(reset)
    values = pd.to_numeric(reset[value_col], errors="coerce")
    idx = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "segment_key": reset["segment_key"].astype(str),
                "origin": reset["origin"].astype(str),
            }
        )
    )
    out = pd.Series(values.to_numpy(dtype=float), index=idx)
    return out.groupby(level=[0, 1]).sum(min_count=1)


def _normalize_summary_index(summary: pd.DataFrame) -> pd.DataFrame:
    if isinstance(summary.index, pd.MultiIndex):
        segs = summary.index.get_level_values(0).unique()
        if len(segs) == 1 and segs[0] == "(All)":
            out = summary.copy()
            out.index = out.index.get_level_values(1)
            out.index.name = "origin"
            return out
    return summary


def _pattern_dict(pattern_triangle: Any | None) -> dict[str, dict[str, float]]:
    if pattern_triangle is None:
        return {}
    frame = _triangle_to_frame(pattern_triangle, origin_as_datetime=False)
    segment_cols = [name for name in frame.index.names if name is not None]
    reset = frame.reset_index()
    reset["segment_key"] = reset.apply(
        lambda row: _segment_key_from_row(row, segment_cols), axis=1
    )
    if "development" not in reset.columns:
        return {}
    value_col = _detect_value_column(reset)
    out: dict[str, dict[str, float]] = {}
    for _, row in reset.iterrows():
        seg = str(row["segment_key"])
        dev = str(row["development"])
        val = pd.to_numeric(row[value_col], errors="coerce")
        if pd.isna(val):
            continue
        out.setdefault(seg, {})[dev] = float(val)
    return out


def _summary_from_model(triangle_input: Any, model: Any) -> pd.DataFrame:
    latest = _triangle_series_by_origin(triangle_input.latest_diagonal)
    ultimate = _triangle_series_by_origin(model.ultimate_)
    ibnr = _triangle_series_by_origin(model.ibnr_)

    idx = latest.index.union(ultimate.index).union(ibnr.index)
    summary = pd.DataFrame(index=idx)
    summary["latest"] = latest.reindex(idx)
    summary["ultimate"] = ultimate.reindex(idx)
    summary["ibnr"] = ibnr.reindex(idx)
    summary["ibnr"] = summary["ibnr"].where(
        summary["ibnr"].notna(), summary["ultimate"] - summary["latest"]
    )
    summary["latest_reported"] = summary["latest"]
    summary = _normalize_summary_index(summary.sort_index())
    return summary


def _method_result(method_name: str, summary: pd.DataFrame, assumption_notes: str) -> MethodResult:
    return MethodResult(
        method_name=method_name,
        latest_reported_total=float(np.nansum(summary["latest"].to_numpy(dtype=float))),
        ultimate_total=float(np.nansum(summary["ultimate"].to_numpy(dtype=float))),
        ibnr_total=float(np.nansum(summary["ibnr"].to_numpy(dtype=float))),
        assumption_notes=assumption_notes,
    )


def _fit_development(
    triangle: Any,
    *,
    average: str | Sequence[str] = "volume",
    n_periods: int = -1,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Fit and apply Development to a Triangle."""

    cl = _import_chainladder()
    dev = cl.Development(average=average, n_periods=n_periods, **kwargs)
    transformed = dev.fit_transform(triangle)
    return transformed, dev


def _apply_tail(
    triangle: Any,
    *,
    tail_kind: str = "constant",
    **tail_kwargs: Any,
) -> tuple[Any, Any]:
    """Apply a tail transformer to a developed Triangle."""

    cl = _import_chainladder()
    if tail_kind == "constant":
        tail_model = cl.TailConstant(**tail_kwargs)
    elif tail_kind == "curve":
        tail_model = cl.TailCurve(**tail_kwargs)
    else:
        raise ValueError("tail_kind must be either 'constant' or 'curve'.")
    transformed = tail_model.fit_transform(triangle)
    return transformed, tail_model


def _prepare_track_a_triangle(
    triangle: Any,
    *,
    development_kwargs: Mapping[str, Any] | None = None,
    tail_kind: str | None = None,
    tail_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    development_kwargs = dict(development_kwargs or {})
    tail_kwargs = dict(tail_kwargs or {})

    dev_triangle, dev_model = _fit_development(triangle, **development_kwargs)
    out_triangle = dev_triangle
    tail_model = None
    if tail_kind is not None:
        out_triangle, tail_model = _apply_tail(
            dev_triangle,
            tail_kind=tail_kind,
            **tail_kwargs,
        )
    return out_triangle, {"development_model": dev_model, "tail_model": tail_model}


def _run_track_a_estimator(
    method_name: str,
    *,
    triangle: Any,
    estimator: Any,
    exposure: pd.Series | None = None,
    assumption_notes: str,
    development_kwargs: Mapping[str, Any] | None = None,
    tail_kind: str | None = None,
    tail_kwargs: Mapping[str, Any] | None = None,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    prepared, prep_artifacts = _prepare_track_a_triangle(
        triangle,
        development_kwargs=development_kwargs,
        tail_kind=tail_kind,
        tail_kwargs=tail_kwargs,
    )
    if exposure is None:
        model = estimator.fit(prepared)
        sample_weight = None
    else:
        sample_weight = build_exposure_triangle(exposure.astype(float), prepared)
        model = estimator.fit(prepared, sample_weight=sample_weight)

    summary = _summary_from_model(prepared, model)
    result = _method_result(method_name, summary, assumption_notes)
    artifacts = {
        "track": "track_a",
        "model": model,
        "prepared_triangle": prepared,
        "development_model": prep_artifacts["development_model"],
        "tail_model": prep_artifacts["tail_model"],
        "sample_weight": sample_weight,
        "selected_ldfs_by_age": _pattern_dict(model.ldf_),
        "selected_cdfs_by_age": _pattern_dict(model.cdf_),
        "tail_assumption": "none" if tail_kind is None else tail_kind,
    }
    return result, summary, artifacts


def run_chain_ladder(
    triangle: Any,
    *,
    development_kwargs: Mapping[str, Any] | None = None,
    tail_kind: str | None = None,
    tail_kwargs: Mapping[str, Any] | None = None,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Track A: deterministic Chain Ladder via chainladder estimator."""

    cl = _import_chainladder()
    return _run_track_a_estimator(
        "Chain Ladder (Track A)",
        triangle=triangle,
        estimator=cl.Chainladder(),
        assumption_notes="Development + optional tail + deterministic Chainladder estimator.",
        development_kwargs=development_kwargs,
        tail_kind=tail_kind,
        tail_kwargs=tail_kwargs,
    )


def run_expected_loss(
    triangle: Any,
    exposure: pd.Series,
    *,
    apriori: float = 1.0,
    development_kwargs: Mapping[str, Any] | None = None,
    tail_kind: str | None = None,
    tail_kwargs: Mapping[str, Any] | None = None,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Expected loss via chainladder ExpectedLoss."""

    cl = _import_chainladder()
    return _run_track_a_estimator(
        "Expected Loss (Track A)",
        triangle=triangle,
        estimator=cl.ExpectedLoss(apriori=apriori),
        exposure=exposure,
        assumption_notes="ExpectedLoss estimator with exposure-based sample weight and apriori.",
        development_kwargs=development_kwargs,
        tail_kind=tail_kind,
        tail_kwargs=tail_kwargs,
    )


def run_bornhuetter_ferguson(
    triangle: Any,
    exposure: pd.Series,
    *,
    apriori: float = 1.0,
    development_kwargs: Mapping[str, Any] | None = None,
    tail_kind: str | None = None,
    tail_kwargs: Mapping[str, Any] | None = None,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Track A: deterministic Bornhuetter-Ferguson via chainladder estimator."""

    cl = _import_chainladder()
    return _run_track_a_estimator(
        "Bornhuetter-Ferguson (Track A)",
        triangle=triangle,
        estimator=cl.BornhuetterFerguson(apriori=apriori),
        exposure=exposure,
        assumption_notes="BornhuetterFerguson estimator with exposure-based sample weight.",
        development_kwargs=development_kwargs,
        tail_kind=tail_kind,
        tail_kwargs=tail_kwargs,
    )


def run_benktander(
    triangle: Any,
    exposure: pd.Series,
    *,
    apriori: float = 1.0,
    n_iters: int = 2,
    development_kwargs: Mapping[str, Any] | None = None,
    tail_kind: str | None = None,
    tail_kwargs: Mapping[str, Any] | None = None,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Track A: deterministic Benktander via chainladder estimator."""

    cl = _import_chainladder()
    return _run_track_a_estimator(
        "Benktander (Track A)",
        triangle=triangle,
        estimator=cl.Benktander(apriori=apriori, n_iters=n_iters),
        exposure=exposure,
        assumption_notes="Benktander estimator with apriori, exposure sample weight, and n_iters.",
        development_kwargs=development_kwargs,
        tail_kind=tail_kind,
        tail_kwargs=tail_kwargs,
    )


def run_cape_cod(
    triangle: Any,
    exposure: pd.Series,
    *,
    trend: float = 0.0,
    decay: float = 1.0,
    n_iters: int = 1,
    development_kwargs: Mapping[str, Any] | None = None,
    tail_kind: str | None = None,
    tail_kwargs: Mapping[str, Any] | None = None,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Track A: deterministic Cape Cod via chainladder estimator."""

    cl = _import_chainladder()
    return _run_track_a_estimator(
        "Cape Cod (Track A)",
        triangle=triangle,
        estimator=cl.CapeCod(trend=trend, decay=decay, n_iters=n_iters),
        exposure=exposure,
        assumption_notes=(
            "CapeCod estimator with exposure sample weight and trend/decay assumptions."
        ),
        development_kwargs=development_kwargs,
        tail_kind=tail_kind,
        tail_kwargs=tail_kwargs,
    )


def run_case_outstanding_chainladder(
    paid_cumulative_triangle: Any,
    incurred_cumulative_triangle: Any,
    *,
    paid_n_periods: int = -1,
    case_n_periods: int = -1,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Case OS pipeline using chainladder.CaseOutstanding development transformer."""

    cl = _import_chainladder()
    paid_name = "paid"
    incurred_name = "incurred"

    paid_tri = paid_cumulative_triangle.copy()
    incurred_tri = incurred_cumulative_triangle.copy()
    paid_tri.columns = [paid_name]
    incurred_tri.columns = [incurred_name]

    joint = cl.concat((paid_tri, incurred_tri), axis=1)
    co = cl.CaseOutstanding(
        paid_to_incurred=(paid_name, incurred_name),
        paid_n_periods=paid_n_periods,
        case_n_periods=case_n_periods,
    ).fit(joint)
    transformed = co.transform(joint)
    model = cl.Chainladder().fit(transformed[incurred_name])

    summary = _summary_from_model(transformed[incurred_name], model)
    result = _method_result(
        "Case Outstanding (chainladder)",
        summary,
        "chainladder CaseOutstanding development transform followed by Chainladder.",
    )
    artifacts = {
        "track": "track_a",
        "case_outstanding_model": co,
        "model": model,
        "prepared_triangle": transformed[incurred_name],
        "selected_ldfs_by_age": {
            "combined": _pattern_dict(co.ldf_),
            "paid_to_prior_case": _pattern_dict(co.paid_ldf_),
            "case_to_prior_case": _pattern_dict(co.case_ldf_),
            "incurred_chainladder": _pattern_dict(model.ldf_),
        },
        "selected_cdfs_by_age": {
            "combined": _pattern_dict(co.cdf_),
            "incurred_chainladder": _pattern_dict(model.cdf_),
        },
        "tail_assumption": "none",
    }
    return result, summary, artifacts




