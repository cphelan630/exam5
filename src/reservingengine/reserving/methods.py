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


def _triangle_to_segment_wide(triangle: Any) -> dict[str, pd.DataFrame]:
    frame = _triangle_to_frame(triangle, origin_as_datetime=False)
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
    count_prepared, count_dev = _fit_development(
        cumulative_count_triangle,
        n_periods=count_n_periods,
    )
    sev_prepared, sev_dev = _fit_development(
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


def _origin_to_year_series(index: pd.Index) -> pd.Series:
    """Convert an origin index to numeric calendar years for trend fitting."""
    if hasattr(index, "year"):
        return pd.Series(index.year, index=index, dtype=float)
    years: list[float] = []
    for v in index:
        try:
            years.append(float(str(v)[:4]))
        except (ValueError, TypeError):
            years.append(float("nan"))
    return pd.Series(years, index=index, dtype=float)


def run_frequency_severity_technique2(
    cumulative_count_triangle: Any,
    severity_triangle: Any,
    exposure: pd.Series,
    *,
    n_mature_origins: int = 5,
    count_n_periods: int = -1,
    severity_n_periods: int = -1,
    latest_reported_claims: pd.Series | None = None,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Track B: Frequency-Severity Technique 2 — trend-projected frequency and severity.

    Friedland Chapter 11, Technique 2:
    1. Develop counts and severities to ultimate (same as Technique 1).
    2. For the ``n_mature_origins`` oldest/most-developed accident years, derive:
         implied frequency = projected ultimate count / exposure
         implied severity  = projected ultimate severity
    3. Fit a log-linear trend to each across accident years.
    4. Project frequency and severity for all years using the fitted trends.
    5. Ultimate claims = exposure × projected frequency × projected severity.

    When recent accident years are immature, development-based ultimates are
    unreliable.  The trend bridges from the stable older years to the newer
    ones, making Technique 2 more stable for the most recent cohorts.
    """
    cl = _import_chainladder()

    # Step 1: Develop counts and severity to ultimate (Technique 1 base)
    count_prepared, count_dev = _fit_development(
        cumulative_count_triangle, n_periods=count_n_periods
    )
    sev_prepared, sev_dev = _fit_development(severity_triangle, n_periods=severity_n_periods)
    count_model = cl.Chainladder().fit(count_prepared)
    sev_model = cl.Chainladder().fit(sev_prepared)

    ultimate_count = _triangle_series_by_origin(count_model.ultimate_)
    ultimate_severity = _triangle_series_by_origin(sev_model.ultimate_)

    idx = ultimate_count.index.union(ultimate_severity.index)
    ult_cnt = ultimate_count.reindex(idx).astype(float)
    ult_sev = ultimate_severity.reindex(idx).astype(float)
    exposure_aligned = _align_external_origin_series(exposure.astype(float), idx)

    # Step 2: Implied frequency from Technique 1 ultimates
    frequency = ult_cnt / exposure_aligned

    # Oldest n_mature_origins rows are most developed
    mature_origins = idx[:n_mature_origins] if len(idx) >= n_mature_origins else idx
    ay_numeric = _origin_to_year_series(idx)

    freq_mature = frequency.loc[mature_origins].dropna()
    sev_mature = ult_sev.loc[mature_origins].dropna()

    # Step 3: Fit log-linear trends; fall back to constant mean if insufficient data
    freq_slope: float | None = None
    sev_slope: float | None = None

    if len(freq_mature) >= 2 and (freq_mature > 0).all():
        freq_b, freq_a = np.polyfit(
            _origin_to_year_series(freq_mature.index).to_numpy(dtype=float),
            np.log(freq_mature.to_numpy(dtype=float)),
            1,
        )
        freq_slope = float(freq_b)
        projected_frequency: pd.Series = pd.Series(
            np.exp(freq_a + freq_b * ay_numeric.to_numpy(dtype=float)), index=idx
        )
    else:
        projected_frequency = pd.Series(float(freq_mature.mean()), index=idx)

    if len(sev_mature) >= 2 and (sev_mature > 0).all():
        sev_b, sev_a = np.polyfit(
            _origin_to_year_series(sev_mature.index).to_numpy(dtype=float),
            np.log(sev_mature.to_numpy(dtype=float)),
            1,
        )
        sev_slope = float(sev_b)
        projected_severity: pd.Series = pd.Series(
            np.exp(sev_a + sev_b * ay_numeric.to_numpy(dtype=float)), index=idx
        )
    else:
        projected_severity = pd.Series(float(sev_mature.mean()), index=idx)

    # Step 4: Ultimate = exposure × projected_frequency × projected_severity
    ultimate_claims = exposure_aligned * projected_frequency * projected_severity

    if latest_reported_claims is not None:
        latest = _align_external_origin_series(latest_reported_claims, idx)
    else:
        latest_count_diag = _triangle_series_by_origin(
            cumulative_count_triangle.latest_diagonal
        )
        latest_sev_diag = _triangle_series_by_origin(severity_triangle.latest_diagonal)
        latest = latest_count_diag.reindex(idx) * latest_sev_diag.reindex(idx)

    ibnr = ultimate_claims - latest
    summary = pd.DataFrame(
        {
            "exposure": exposure_aligned,
            "projected_frequency": projected_frequency,
            "projected_severity": projected_severity,
            "ultimate": ultimate_claims,
            "latest": latest,
            "ibnr": ibnr,
        },
        index=idx,
    )
    summary = _normalize_summary_index(summary.sort_index())
    result = _method_result(
        "Frequency-Severity Technique 2 (Track B)",
        summary,
        (
            f"Trend-projected frequency and severity applied to exposure. "
            f"Trend fitted from {n_mature_origins} most mature origin years."
        ),
    )
    artifacts = {
        "track": "track_b",
        "count_model": count_model,
        "severity_model": sev_model,
        "count_development_model": count_dev,
        "severity_development_model": sev_dev,
        "frequency_trend_slope": freq_slope,
        "severity_trend_slope": sev_slope,
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


def run_disposal_rate(
    closed_count_triangle: Any,
    paid_claims_triangle: Any,
    disposal_rate_triangle: Any,
    *,
    count_n_periods: int = -1,
    n_tail_ages: int = 3,
    dr_n_periods: int = 5,
) -> tuple[MethodResult, pd.DataFrame, dict[str, Any]]:
    """Track B: Disposal Rate method (Friedland Chapter 13).

    The disposal rate DR(w, d) = cumulative closed claims at age d for origin w
    divided by ultimate closed claims for that origin.  It represents the
    proportion of ultimate closed claims settled by a given development age.

    Steps:
    1. Develop cumulative closed claim counts to ultimate using Development +
       Chainladder to obtain ultimate closed counts by origin year.
    2. For each development age, select a disposal rate using the simple average
       of the most recent ``dr_n_periods`` observed values.  Enforce monotone
       increase and cap at 1.0; set the final age to exactly 1.0.
    3. Compute incremental severity at each age = incremental paid / incremental
       closed counts.  For the last ``n_tail_ages`` development ages, use their
       mean as a stable tail severity assumption.
    4. For each origin year at its current latest age, project future paid
       claims as the sum over future ages of:
         projected incremental closed × selected incremental severity.
    5. Ultimate paid = latest paid + projected future paid.
    """
    cl = _import_chainladder()

    # Step 1: Develop closed counts to ultimate
    count_prepared, count_dev = _fit_development(
        closed_count_triangle, n_periods=count_n_periods
    )
    count_model = cl.Chainladder().fit(count_prepared)
    ultimate_closed = _triangle_series_by_origin(count_model.ultimate_)

    # Build wide matrices
    dr_wide = _triangle_to_segment_wide(disposal_rate_triangle)
    closed_wide = _triangle_to_segment_wide(closed_count_triangle)
    paid_wide = _triangle_to_segment_wide(paid_claims_triangle)

    # Use only the "(All)" segment (single-segment triangles)
    seg = next(iter(dr_wide))
    dr_mat = dr_wide[seg]
    closed_mat = closed_wide.get(seg, pd.DataFrame())
    paid_mat = paid_wide.get(seg, pd.DataFrame())

    age_cols = dr_mat.columns.tolist()
    try:
        age_cols_num = [int(a) for a in age_cols]
    except (ValueError, TypeError):
        age_cols_num = list(range(len(age_cols)))

    # Step 2: Selected disposal rates per age
    selected_dr: dict[Any, float] = {}
    prev_dr = 0.0
    for age in age_cols:
        vals = dr_mat[age].dropna()
        recent = vals.tail(dr_n_periods) if len(vals) >= dr_n_periods else vals
        raw_dr = float(recent.mean()) if len(recent) > 0 else prev_dr
        # Enforce monotone increase
        selected_dr[age] = max(raw_dr, prev_dr)
        prev_dr = selected_dr[age]
    # Final age = 1.0 (fully developed)
    if age_cols:
        selected_dr[age_cols[-1]] = 1.0

    # Step 3: Incremental severity per age
    inc_paid = paid_mat.diff(axis=1).copy()
    inc_paid.iloc[:, 0] = paid_mat.iloc[:, 0]
    inc_closed = closed_mat.diff(axis=1).copy()
    inc_closed.iloc[:, 0] = closed_mat.iloc[:, 0]
    inc_sev_mat = (inc_paid / inc_closed).replace([np.inf, -np.inf], np.nan)

    selected_sev: dict[Any, float] = {}
    for age in age_cols:
        if age in inc_sev_mat.columns:
            vals = inc_sev_mat[age].replace([np.inf, -np.inf], np.nan).dropna()
            recent = vals.tail(dr_n_periods) if len(vals) >= dr_n_periods else vals
            selected_sev[age] = float(recent.median()) if len(recent) > 0 else np.nan
        else:
            selected_sev[age] = np.nan

    # Tail severity: average over last n_tail_ages
    tail_ages = age_cols[-n_tail_ages:] if len(age_cols) >= n_tail_ages else age_cols
    tail_sev_vals = [selected_sev[a] for a in tail_ages if not np.isnan(selected_sev.get(a, np.nan))]
    tail_sev = float(np.mean(tail_sev_vals)) if tail_sev_vals else np.nan
    for age in tail_ages:
        selected_sev[age] = tail_sev

    # Forward-fill any remaining NaN severities
    sev_series = pd.Series(selected_sev)
    sev_series = sev_series.ffill()

    # Step 4: Project future paid for each origin
    ultimate_closed_map = {str(k): float(v) for k, v in ultimate_closed.items()}

    summary_rows: list[dict[str, Any]] = []
    for origin in dr_mat.index:
        origin_str = str(origin)
        u_closed = ultimate_closed_map.get(origin_str, np.nan)
        if np.isnan(u_closed):
            continue

        # Latest age for this origin
        valid_ages = dr_mat.loc[origin].dropna().index.tolist()
        if not valid_ages:
            continue
        latest_age_label = valid_ages[-1]

        # Latest paid (cumulative)
        if latest_age_label in paid_mat.columns and pd.notna(paid_mat.at[origin, latest_age_label]):
            latest_paid_val = float(paid_mat.at[origin, latest_age_label])
        else:
            valid_paid = paid_mat.loc[origin].dropna()
            latest_paid_val = float(valid_paid.iloc[-1]) if len(valid_paid) > 0 else 0.0

        # DR at latest known age
        dr_at_latest = selected_dr.get(latest_age_label, 0.0)

        # Project future ages
        future_paid = 0.0
        past_latest = False
        prev_dr_val = dr_at_latest
        for age in age_cols:
            if not past_latest:
                if age == latest_age_label:
                    past_latest = True
                continue
            dr_age = selected_dr.get(age, prev_dr_val)
            inc_closed_projected = max(u_closed * (dr_age - prev_dr_val), 0.0)
            inc_sev = sev_series.get(age, tail_sev)
            if not np.isnan(inc_sev):
                future_paid += inc_closed_projected * inc_sev
            prev_dr_val = dr_age

        ultimate_paid = latest_paid_val + future_paid
        ibnr = ultimate_paid - latest_paid_val
        summary_rows.append(
            {
                "origin": origin_str,
                "latest_age": latest_age_label,
                "ultimate_closed_counts": u_closed,
                "disposal_rate_at_latest": dr_at_latest,
                "latest": latest_paid_val,
                "ultimate": ultimate_paid,
                "ibnr": ibnr,
            }
        )

    summary = (
        pd.DataFrame(summary_rows).set_index("origin").sort_index()
        if summary_rows
        else pd.DataFrame(columns=["latest", "ultimate", "ibnr"])
    )
    result = _method_result(
        "Disposal Rate (Track B)",
        summary,
        (
            "Disposal rate method: project future paid via incremental closed counts "
            "× incremental severity at each age."
        ),
    )
    selected_dr_series = pd.Series(selected_dr)
    selected_sev_series = sev_series
    artifacts = {
        "track": "track_b",
        "count_model": count_model,
        "count_development_model": count_dev,
        "selected_disposal_rates": selected_dr_series.to_dict(),
        "selected_incremental_severities": selected_sev_series.to_dict(),
        "selected_ldfs_by_age": {"closed_counts": _pattern_dict(count_model.ldf_)},
        "selected_cdfs_by_age": {"closed_counts": _pattern_dict(count_model.cdf_)},
        "tail_assumption": "implicit_via_dr_1.0",
    }
    return result, summary, artifacts


