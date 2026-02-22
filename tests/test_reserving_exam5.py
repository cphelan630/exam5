from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("chainladder")

from reservingengine.reserving import (
    build_reconciliation_report,
    build_triangle,
    compare_patterns_to_baseline,
    compare_results_to_baseline,
    load_baseline_fixture,
    load_raw_dataset,
    run_benktander,
    run_bornhuetter_ferguson,
    run_cape_cod,
    run_case_outstanding_chainladder,
    run_case_outstanding_friedland,
    run_chain_ladder,
    run_frequency_severity_friedland,
    snapshot_method_output,
    validate_cum_incr_roundtrip,
    validate_triangle_totals_match_raw,
)

BASELINE_PATH = Path("tests/fixtures/reserving_exam5_baseline.json")


def _origin_exposure(
    triangle: object,
    *,
    start: float = 1_000_000.0,
    step: float = 50_000.0,
) -> pd.Series:
    latest = triangle.latest_diagonal.to_frame(origin_as_datetime=False, keepdims=True)
    origins = pd.Index(latest["origin"])
    values = start + step * np.arange(len(origins), dtype=float)
    return pd.Series(values, index=origins, dtype=float)


def _build_current_snapshots() -> dict[str, dict]:
    raw = load_raw_dataset("friedland_us_industry_auto")
    triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Reported Claims"],
        cumulative=True,
    )
    exposure = _origin_exposure(triangle)
    development_kwargs = {"average": "volume", "n_periods": -1}

    current: dict[str, dict] = {}
    result, summary, artifacts = run_chain_ladder(
        triangle,
        development_kwargs=development_kwargs,
    )
    current[result.method_name] = snapshot_method_output(result, summary, artifacts)
    result, summary, artifacts = run_bornhuetter_ferguson(
        triangle,
        exposure,
        apriori=0.70,
        development_kwargs=development_kwargs,
    )
    current[result.method_name] = snapshot_method_output(result, summary, artifacts)
    result, summary, artifacts = run_benktander(
        triangle,
        exposure,
        apriori=0.70,
        n_iters=3,
        development_kwargs=development_kwargs,
    )
    current[result.method_name] = snapshot_method_output(result, summary, artifacts)
    result, summary, artifacts = run_cape_cod(
        triangle,
        exposure,
        trend=0.0,
        decay=1.0,
        n_iters=1,
        development_kwargs=development_kwargs,
    )
    current[result.method_name] = snapshot_method_output(result, summary, artifacts)

    case_raw = load_raw_dataset("friedland_us_industry_auto_case")
    case_triangle = build_triangle(
        case_raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Case Outstanding"],
        cumulative=False,
    )
    paid_inc_triangle = build_triangle(
        case_raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Incremental Paid Claims"],
        cumulative=False,
    )
    result, summary, artifacts = run_case_outstanding_friedland(case_triangle, paid_inc_triangle)
    current[result.method_name] = snapshot_method_output(result, summary, artifacts)

    case_cum = case_raw.copy()
    case_cum["Paid Cumulative"] = case_cum.groupby("Accident Year")[
        "Incremental Paid Claims"
    ].cumsum()
    case_cum["Incurred Cumulative"] = case_cum["Paid Cumulative"] + case_cum["Case Outstanding"]
    paid_cum_triangle = build_triangle(
        case_cum,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Paid Cumulative"],
        cumulative=True,
    )
    incurred_cum_triangle = build_triangle(
        case_cum,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Incurred Cumulative"],
        cumulative=True,
    )
    result, summary, artifacts = run_case_outstanding_chainladder(
        paid_cum_triangle,
        incurred_cum_triangle,
    )
    current[result.method_name] = snapshot_method_output(result, summary, artifacts)

    freq_raw = load_raw_dataset("friedland_auto_freq_sev")
    count_triangle = build_triangle(
        freq_raw,
        origin_col="Accident Half-Year",
        development_col="Calendar Half-Year",
        value_cols=["Reported Claim Counts"],
        cumulative=True,
    )
    sev_triangle = build_triangle(
        freq_raw,
        origin_col="Accident Half-Year",
        development_col="Calendar Half-Year",
        value_cols=["Reported Severity"],
        cumulative=True,
    )
    result, summary, artifacts = run_frequency_severity_friedland(count_triangle, sev_triangle)
    current[result.method_name] = snapshot_method_output(result, summary, artifacts)
    return current


def test_triangle_totals_match_raw_data() -> None:
    raw = load_raw_dataset("friedland_us_industry_auto")
    triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Reported Claims"],
        cumulative=True,
    )
    ok, raw_total, tri_total = validate_triangle_totals_match_raw(
        raw,
        triangle,
        raw_value_cols=["Reported Claims"],
    )
    assert ok
    assert raw_total == pytest.approx(tri_total)


def test_cumulative_incremental_roundtrip_consistent() -> None:
    raw = load_raw_dataset("friedland_us_industry_auto")
    triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Reported Claims"],
        cumulative=True,
    )
    ok, max_abs_diff = validate_cum_incr_roundtrip(triangle)
    assert ok
    assert max_abs_diff <= 1e-8


def test_selected_ldfs_match_baseline() -> None:
    baseline = load_baseline_fixture(BASELINE_PATH)["methods"]
    current = _build_current_snapshots()
    compare = compare_patterns_to_baseline(current, baseline)
    assert not compare.empty
    assert bool(compare["passed"].all())


def test_ultimates_by_ay_match_baseline() -> None:
    baseline = load_baseline_fixture(BASELINE_PATH)["methods"]
    current = _build_current_snapshots()
    compare = compare_results_to_baseline(current, baseline)
    ay_ultimate = compare[compare["metric_group"] == "ultimate_by_ay"]
    assert not ay_ultimate.empty
    assert bool(ay_ultimate["passed"].all())


def test_ibnr_by_ay_and_latest_diagonal_match_baseline() -> None:
    baseline = load_baseline_fixture(BASELINE_PATH)["methods"]
    current = _build_current_snapshots()
    compare = compare_results_to_baseline(current, baseline)
    ibnr_rows = compare[compare["metric_group"] == "ibnr_by_ay"]
    latest_rows = compare[compare["metric_group"] == "latest_diagonal_by_ay"]
    assert not ibnr_rows.empty
    assert not latest_rows.empty
    assert bool(ibnr_rows["passed"].all())
    assert bool(latest_rows["passed"].all())


def test_sparse_triangle_edge_case_is_stable() -> None:
    raw = load_raw_dataset("friedland_xyz_auto_bi")
    triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Reported Claims"],
        cumulative=True,
    )
    result, summary, _ = run_chain_ladder(triangle)
    assert np.isfinite(result.ultimate_total)
    assert np.isfinite(result.latest_reported_total)
    assert "ultimate" in summary.columns
    assert "ibnr" in summary.columns


def test_track_b_case_os_ratio_and_projection_sanity() -> None:
    raw = load_raw_dataset("friedland_us_industry_auto_case")
    case_triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Case Outstanding"],
        cumulative=False,
    )
    paid_inc_triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Incremental Paid Claims"],
        cumulative=False,
    )
    result, summary, artifacts = run_case_outstanding_friedland(case_triangle, paid_inc_triangle)
    ratio_table = artifacts["ratio_table"]
    assert not ratio_table.empty
    assert {"paid_to_prior_case", "case_to_prior_case"}.issubset(ratio_table.columns)
    assert result.ultimate_total >= result.latest_reported_total
    assert bool((summary["ultimate"] >= summary["latest"]).all())


def test_case_os_chainladder_path_smoke_and_comparison() -> None:
    raw = load_raw_dataset("friedland_us_industry_auto_case")
    case_triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Case Outstanding"],
        cumulative=False,
    )
    paid_inc_triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Incremental Paid Claims"],
        cumulative=False,
    )
    friedland_result, _, _ = run_case_outstanding_friedland(case_triangle, paid_inc_triangle)

    raw["Paid Cumulative"] = raw.groupby("Accident Year")["Incremental Paid Claims"].cumsum()
    raw["Incurred Cumulative"] = raw["Paid Cumulative"] + raw["Case Outstanding"]
    paid_cum_triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Paid Cumulative"],
        cumulative=True,
    )
    incurred_cum_triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Incurred Cumulative"],
        cumulative=True,
    )
    chain_result, chain_summary, chain_artifacts = run_case_outstanding_chainladder(
        paid_cum_triangle,
        incurred_cum_triangle,
    )
    assert np.isfinite(chain_result.ultimate_total)
    assert np.isfinite(chain_result.latest_reported_total)
    assert {"combined", "paid_to_prior_case", "case_to_prior_case"}.issubset(
        chain_artifacts["selected_ldfs_by_age"].keys()
    )
    assert "ultimate" in chain_summary.columns
    # They should be directionally similar but not identical implementations.
    rel_diff = abs(chain_result.ultimate_total - friedland_result.ultimate_total) / max(
        abs(friedland_result.ultimate_total),
        1.0,
    )
    assert rel_diff < 0.25


def test_track_b_frequency_severity_reconstruction() -> None:
    raw = load_raw_dataset("friedland_auto_freq_sev")
    count_triangle = build_triangle(
        raw,
        origin_col="Accident Half-Year",
        development_col="Calendar Half-Year",
        value_cols=["Reported Claim Counts"],
        cumulative=True,
    )
    sev_triangle = build_triangle(
        raw,
        origin_col="Accident Half-Year",
        development_col="Calendar Half-Year",
        value_cols=["Reported Severity"],
        cumulative=True,
    )
    result, summary, _ = run_frequency_severity_friedland(count_triangle, sev_triangle)
    assert np.isfinite(result.ultimate_total)
    assert np.isfinite(result.latest_reported_total)
    assert np.allclose(
        summary["ultimate_claims"].to_numpy(dtype=float),
        (summary["ultimate_count"] * summary["ultimate_severity"]).to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-8,
        equal_nan=True,
    )


def test_reconciliation_report_has_no_failures() -> None:
    baseline = load_baseline_fixture(BASELINE_PATH)["methods"]
    current = _build_current_snapshots()
    report = build_reconciliation_report(current, baseline)
    assert not report.empty
    assert bool(report["passed"].all())
