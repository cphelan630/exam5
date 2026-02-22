# Exam 5 reserving module docs

This project now uses a two-track reserving architecture.

## Track A (chainladder-native)

Module: `src/reservingengine/reserving/methods.py`

- `fit_development(...)`
- `apply_tail(...)`
- `run_chain_ladder(...)`
- `run_bornhuetter_ferguson(...)`
- `run_benktander(...)`
- `run_cape_cod(...)`

These methods use `chainladder.Triangle` + chainladder estimators directly.

## Track B (Friedland techniques implemented in-project)

Module: `src/reservingengine/reserving/methods.py`

- `run_case_outstanding_friedland(...)`
- `run_case_outstanding_chainladder(...)`
- `run_frequency_severity_friedland(...)`

These methods keep Friedland mechanics explicit while still using `Triangle`
objects for data handling where helpful.

For Case OS you now have both paths:
- `run_case_outstanding_friedland(...)`: textbook-style transparent ratio mechanics
- `run_case_outstanding_chainladder(...)`: chainladder `CaseOutstanding` transform
  + downstream chain ladder pipeline

## Triangle IO

Module: `src/reservingengine/reserving/triangle_io.py`

- `load_raw_dataset(...)`
- `build_triangle(...)`
- `build_exposure_triangle(...)`
- `validate_triangle_totals_match_raw(...)`
- `validate_cum_incr_roundtrip(...)`

## Diagnostics

Module: `src/reservingengine/reserving/diagnostics.py`

- `triangle_sanity_checks(...)`
- `link_ratio_table(...)`
- `selected_ldf_table(...)`
- `run_correlation_tests(...)`
- `plot_link_ratio_heatmap(...)`

## Reconciliation / Baselines

Module: `src/reservingengine/reserving/compare.py`

- `snapshot_method_output(...)`
- `load_baseline_fixture(...)`
- `save_baseline_fixture(...)`
- `compare_results_to_baseline(...)`
- `compare_patterns_to_baseline(...)`
- `build_reconciliation_report(...)`

Baseline fixture:
- `tests/fixtures/reserving_exam5_baseline.json`

Regenerate baseline explicitly:

```powershell
$env:PYTHONPATH='src'
.\.venv_exam5\Scripts\python scripts\generate_reserving_exam5_baseline.py
```

This project intentionally uses the new Track A / Track B modules directly.
