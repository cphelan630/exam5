"""Generate pattern-level baseline fixture for Exam 5 reserving methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservingengine.reserving import (
    build_triangle,
    run_benktander,
    run_bornhuetter_ferguson,
    run_cape_cod,
    run_case_outstanding_chainladder,
    run_case_outstanding_friedland,
    run_chain_ladder,
    run_frequency_severity_friedland,
    save_baseline_fixture,
    snapshot_method_output,
)
from reservingengine.reserving.triangle_io import (
    default_chainladder_data_dir,
    load_raw_dataset,
)


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


def _build_track_a_snapshots(data_dir: Path) -> dict[str, dict]:
    raw = load_raw_dataset("friedland_us_industry_auto", data_dir=data_dir)
    triangle = build_triangle(
        raw,
        origin_col="Accident Year",
        development_col="Calendar Year",
        value_cols=["Reported Claims"],
        cumulative=True,
    )
    exposure = _origin_exposure(triangle)
    development_kwargs = {"average": "volume", "n_periods": -1}

    snapshots: dict[str, dict] = {}
    result, summary, artifacts = run_chain_ladder(
        triangle,
        development_kwargs=development_kwargs,
    )
    snapshots[result.method_name] = snapshot_method_output(
        result,
        summary,
        artifacts,
        metadata={"dataset": "friedland_us_industry_auto"},
    )
    result, summary, artifacts = run_bornhuetter_ferguson(
        triangle,
        exposure,
        apriori=0.70,
        development_kwargs=development_kwargs,
    )
    snapshots[result.method_name] = snapshot_method_output(
        result,
        summary,
        artifacts,
        metadata={"dataset": "friedland_us_industry_auto"},
    )
    result, summary, artifacts = run_benktander(
        triangle,
        exposure,
        apriori=0.70,
        n_iters=3,
        development_kwargs=development_kwargs,
    )
    snapshots[result.method_name] = snapshot_method_output(
        result,
        summary,
        artifacts,
        metadata={"dataset": "friedland_us_industry_auto"},
    )
    result, summary, artifacts = run_cape_cod(
        triangle,
        exposure,
        trend=0.0,
        decay=1.0,
        n_iters=1,
        development_kwargs=development_kwargs,
    )
    snapshots[result.method_name] = snapshot_method_output(
        result,
        summary,
        artifacts,
        metadata={"dataset": "friedland_us_industry_auto"},
    )
    return snapshots


def _build_track_b_snapshots(data_dir: Path) -> dict[str, dict]:
    snapshots: dict[str, dict] = {}

    case_raw = load_raw_dataset("friedland_us_industry_auto_case", data_dir=data_dir)
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
    snapshots[result.method_name] = snapshot_method_output(
        result,
        summary,
        artifacts,
        metadata={"dataset": "friedland_us_industry_auto_case"},
    )

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
    snapshots[result.method_name] = snapshot_method_output(
        result,
        summary,
        artifacts,
        metadata={"dataset": "friedland_us_industry_auto_case"},
    )

    freq_raw = load_raw_dataset("friedland_auto_freq_sev", data_dir=data_dir)
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
    snapshots[result.method_name] = snapshot_method_output(
        result,
        summary,
        artifacts,
        metadata={"dataset": "friedland_auto_freq_sev"},
    )
    return snapshots


def main() -> None:
    data_dir = default_chainladder_data_dir()
    payload = {
        "version": "2.0",
        "metadata": {
            "generator": "scripts/generate_reserving_exam5_baseline.py",
            "track_a_tail_default": "none",
        },
        "methods": {},
    }
    payload["methods"].update(_build_track_a_snapshots(data_dir))
    payload["methods"].update(_build_track_b_snapshots(data_dir))
    out_path = Path("tests") / "fixtures" / "reserving_exam5_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_baseline_fixture(out_path, payload)
    print(f"Wrote baseline fixture: {out_path}")


if __name__ == "__main__":
    main()
