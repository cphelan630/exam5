"""Reserving methods split into Track A and Track B workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .triangle_io import build_exposure_triangle, triangle_to_frame

_META_COLUMNS = {"origin", "development", "valuation"}


@dataclass
class MethodResult:
    """Compact summary for one reserving method run."""

    method_name: str
    ultimate_total: float
    ibnr_total: float
    latest_reported_total: float
    assumption_notes: str


def _import_chainladder() -> Any:
    try:
        import chainladder as cl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "chainladder is required for this method. Install it in your exam5 environment."
        ) from exc
    return cl


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
    frame = triangle_to_frame(triangle, origin_as_datetime=False)
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
    frame = triangle_to_frame(pattern_triangle, origin_as_datetime=False)
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


def fit_development(
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


def apply_tail(
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

    dev_triangle, dev_model = fit_development(triangle, **development_kwargs)
    out_triangle = dev_triangle
    tail_model = None
    if tail_kind is not None:
        out_triangle, tail_model = apply_tail(
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


def _triangle_to_segment_wide(triangle: Any) -> dict[str, pd.DataFrame]:
    frame = triangle_to_frame(triangle, origin_as_datetime=False)
    segment_cols = [name for name in frame.index.names if name is not None]
    reset = frame.reset_index()
    reset["segment_key"] = reset.apply(
        lambda row: _segment_key_from_row(row, segment_cols), axis=1
    )
    if "origin" not in reset.columns or "development" not in reset.columns:
        raise ValueError("Triangle frame must include origin and development columns.")
    value_col = _detect_value_column(reset)
    out: dict[str, pd.DataFrame] = {}
    for segment_key, grp in reset.groupby("segment_key", dropna=False):
        wide = (
            grp.pivot(index="origin", columns="development", values=value_col)
            .sort_index()
            .sort_index(axis=1)
        )
        out[str(segment_key)] = wide.astype(float)
    return out


def run_case_outstanding_friedland(
    case_outstanding_triangle: Any,
    incremental_paid_triangle: Any,
    *,
    n_periods: int | None = None,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Track B: Friedland-style Case OS projection using Triangle containers."""

    case_wide = _triangle_to_segment_wide(case_outstanding_triangle)
    paid_inc_wide = _triangle_to_segment_wide(incremental_paid_triangle)

    summary_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    selected_ldfs: dict[str, dict[str, dict[str, float]]] = {}

    for segment_key, case_tri in case_wide.items():
        if segment_key not in paid_inc_wide:
            raise ValueError(f"Missing paid triangle segment: {segment_key}")
        paid_inc = paid_inc_wide[segment_key].reindex(case_tri.index).reindex(
            case_tri.columns, axis=1
        )
        paid_cum = paid_inc.cumsum(axis=1)

        if n_periods is not None and 0 < n_periods < len(case_tri):
            case_fit = case_tri.tail(n_periods)
            paid_fit = paid_inc.tail(n_periods)
        else:
            case_fit = case_tri
            paid_fit = paid_inc

        ages = list(case_tri.columns)
        paid_ratio_by_age: dict[Any, float] = {}
        case_ratio_by_age: dict[Any, float] = {}
        selected_ldfs[segment_key] = {"paid_to_prior_case": {}, "case_to_prior_case": {}}

        for i in range(len(ages) - 1):
            age = ages[i]
            next_age = ages[i + 1]
            den = case_fit[age]
            case_num = case_fit[next_age]
            paid_num = paid_fit[next_age]

            valid_case = den.notna() & case_num.notna() & (den != 0)
            valid_paid = den.notna() & paid_num.notna() & (den != 0)

            case_ratio = (
                float(case_num[valid_case].sum()) / float(den[valid_case].sum())
                if valid_case.any()
                else np.nan
            )
            paid_ratio = (
                float(paid_num[valid_paid].sum()) / float(den[valid_paid].sum())
                if valid_paid.any()
                else np.nan
            )
            paid_ratio_by_age[age] = paid_ratio
            case_ratio_by_age[age] = case_ratio
            selected_ldfs[segment_key]["paid_to_prior_case"][str(age)] = float(paid_ratio)
            selected_ldfs[segment_key]["case_to_prior_case"][str(age)] = float(case_ratio)
            ratio_rows.append(
                {
                    "segment_key": segment_key,
                    "age": age,
                    "next_age": next_age,
                    "paid_to_prior_case": paid_ratio,
                    "case_to_prior_case": case_ratio,
                }
            )

        latest_case = case_tri.apply(
            lambda row: row.dropna().iloc[-1] if row.notna().any() else np.nan, axis=1
        )
        latest_age = case_tri.apply(
            lambda row: row.dropna().index[-1] if row.notna().any() else np.nan, axis=1
        )
        latest_paid = paid_cum.apply(
            lambda row: row.dropna().iloc[-1] if row.notna().any() else np.nan, axis=1
        )

        for origin in case_tri.index:
            cur_age = latest_age.loc[origin]
            if pd.isna(cur_age):
                continue
            case_balance = float(latest_case.loc[origin])
            future_paid = 0.0
            start_idx = ages.index(cur_age)
            for j in range(start_idx, len(ages) - 1):
                age = ages[j]
                p_ratio = paid_ratio_by_age.get(age, np.nan)
                c_ratio = case_ratio_by_age.get(age, np.nan)
                p_ratio = 0.0 if np.isnan(p_ratio) else p_ratio
                c_ratio = 0.0 if np.isnan(c_ratio) else c_ratio
                paid_amt = case_balance * p_ratio
                case_balance = case_balance * c_ratio
                future_paid += paid_amt
            future_paid += case_balance

            if pd.notna(latest_paid.loc[origin]):
                latest_paid_val = float(latest_paid.loc[origin])
            else:
                latest_paid_val = 0.0
            ultimate = latest_paid_val + future_paid
            ibnr = ultimate - latest_paid_val
            summary_rows.append(
                {
                    "segment_key": segment_key,
                    "origin": str(origin),
                    "latest_age": cur_age,
                    "latest_case_outstanding": float(latest_case.loc[origin]),
                    "latest_paid_cumulative": latest_paid_val,
                    "future_paid": future_paid,
                    "latest": latest_paid_val,
                    "ultimate": ultimate,
                    "ibnr": ibnr,
                }
            )

    summary = pd.DataFrame(summary_rows).set_index(["segment_key", "origin"]).sort_index()
    summary = _normalize_summary_index(summary)
    ratios_df = pd.DataFrame(ratio_rows)
    result = _method_result(
        "Case Outstanding (Track B)",
        summary,
        (
            "Friedland-style Case OS projection using paid-to-prior-case "
            "and case-to-prior-case ratios."
        ),
    )
    artifacts = {
        "track": "track_b",
        "selected_ldfs_by_age": selected_ldfs,
        "selected_cdfs_by_age": {},
        "ratio_table": ratios_df,
        "tail_assumption": "implicit_case_release",
    }
    return result, summary, artifacts


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


def _align_external_origin_series(
    values: pd.Series,
    target_index: pd.Index | pd.MultiIndex,
) -> pd.Series:
    if not isinstance(target_index, pd.MultiIndex):
        aligned = values.reindex(target_index)
        if aligned.isna().any():
            str_values = values.copy()
            str_values.index = str_values.index.map(str)
            aligned = str_values.reindex(target_index.map(str))
        return aligned.astype(float)

    origins = target_index.get_level_values(-1)
    aligned = values.reindex(origins)
    if aligned.isna().any():
        str_values = values.copy()
        str_values.index = str_values.index.map(str)
        aligned = str_values.reindex(origins.map(str))
    out = pd.Series(aligned.to_numpy(dtype=float), index=target_index)
    return out


def run_frequency_severity_friedland(
    cumulative_count_triangle: Any,
    severity_triangle: Any,
    *,
    latest_reported_claims: pd.Series | None = None,
    count_n_periods: int = -1,
    severity_n_periods: int = -1,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Track B: Frequency-Severity using Triangle containers plus custom recombination."""

    cl = _import_chainladder()
    count_prepared, count_dev = fit_development(
        cumulative_count_triangle,
        n_periods=count_n_periods,
    )
    sev_prepared, sev_dev = fit_development(
        severity_triangle,
        n_periods=severity_n_periods,
    )

    count_model = cl.Chainladder().fit(count_prepared)
    sev_model = cl.Chainladder().fit(sev_prepared)

    latest_count = _triangle_series_by_origin(cumulative_count_triangle.latest_diagonal)
    latest_severity = _triangle_series_by_origin(severity_triangle.latest_diagonal)
    ultimate_count = _triangle_series_by_origin(count_model.ultimate_)
    ultimate_severity = _triangle_series_by_origin(sev_model.ultimate_)

    idx = latest_count.index.union(latest_severity.index).union(ultimate_count.index).union(
        ultimate_severity.index
    )
    summary = pd.DataFrame(index=idx)
    summary["latest_count"] = latest_count.reindex(idx)
    summary["latest_severity"] = latest_severity.reindex(idx)
    summary["ultimate_count"] = ultimate_count.reindex(idx)
    summary["ultimate_severity"] = ultimate_severity.reindex(idx)
    summary["ultimate"] = summary["ultimate_count"] * summary["ultimate_severity"]

    if latest_reported_claims is None:
        summary["latest"] = summary["latest_count"] * summary["latest_severity"]
    else:
        summary["latest"] = _align_external_origin_series(latest_reported_claims, summary.index)
    summary["ibnr"] = summary["ultimate"] - summary["latest"]
    summary["ultimate_claims"] = summary["ultimate"]
    summary["latest_reported_claims"] = summary["latest"]
    summary = _normalize_summary_index(summary.sort_index())

    result = _method_result(
        "Frequency-Severity (Track B)",
        summary,
        "Projected ultimate counts and severities are combined into ultimate claims.",
    )
    artifacts = {
        "track": "track_b",
        "count_model": count_model,
        "severity_model": sev_model,
        "count_development_model": count_dev,
        "severity_development_model": sev_dev,
        "selected_ldfs_by_age": {
            "count": _pattern_dict(count_model.ldf_),
            "severity": _pattern_dict(sev_model.ldf_),
        },
        "selected_cdfs_by_age": {
            "count": _pattern_dict(count_model.cdf_),
            "severity": _pattern_dict(sev_model.cdf_),
        },
        "tail_assumption": "none",
    }
    return result, summary, artifacts


def compare_method_outputs(
    results: Sequence[MethodResult],
    *,
    baseline_method: str | None = None,
) -> pd.DataFrame:
    """Build a side-by-side method comparison table."""

    rows = [
        {
            "method_name": r.method_name,
            "latest_reported_total": r.latest_reported_total,
            "ultimate_total": r.ultimate_total,
            "ibnr_total": r.ibnr_total,
            "assumption_notes": r.assumption_notes,
        }
        for r in results
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    baseline_name = baseline_method or out.iloc[0]["method_name"]
    base = out.loc[out["method_name"] == baseline_name, "ultimate_total"]
    if len(base) == 1:
        base_val = float(base.iloc[0])
        out["delta_vs_baseline"] = out["ultimate_total"] - base_val
        out["pct_vs_baseline"] = np.where(
            base_val != 0.0,
            out["delta_vs_baseline"] / base_val,
            np.nan,
        )
    else:
        out["delta_vs_baseline"] = np.nan
        out["pct_vs_baseline"] = np.nan
    return out
