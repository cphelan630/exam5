import pandas as pd

import reservingengine
from reservingengine import MethodResult, snapshot_method_output


def test_public_api_exports() -> None:
    expected = {
        "DEFAULT_TOLERANCES",
        "MethodResult",
        "build_exposure_triangle",
        "build_reconciliation_report",
        "calendar_year_diagnostic",
        "compare_patterns_to_baseline",
        "compare_results_to_baseline",
        "link_ratio_table",
        "paid_vs_incurred_comparison",
        "plot_link_ratio_heatmap",
        "run_benktander",
        "run_bornhuetter_ferguson",
        "run_cape_cod",
        "run_case_outstanding_chainladder",
        "run_chain_ladder",
        "run_expected_loss",
        "snapshot_method_output",
        "trend_summary",
    }
    assert expected.issubset(set(reservingengine.__all__))


def test_snapshot_method_output_smoke() -> None:
    summary = pd.DataFrame(
        {
            "latest": [100.0, 140.0],
            "ultimate": [125.0, 165.0],
            "ibnr": [25.0, 25.0],
        },
        index=pd.Index(["2022", "2023"], name="origin"),
    )
    result = MethodResult(
        method_name="Smoke Method",
        latest_reported_total=240.0,
        ultimate_total=290.0,
        ibnr_total=50.0,
        assumption_notes="Smoke test payload",
    )

    payload = snapshot_method_output(result, summary)

    assert payload["method_name"] == "Smoke Method"
    assert payload["totals"] == {
        "latest": 240.0,
        "ultimate": 290.0,
        "ibnr": 50.0,
    }
    assert payload["latest_diagonal_by_ay"]["(All)"]["2022"] == 100.0
    assert payload["ultimate_by_ay"]["(All)"]["2023"] == 165.0
