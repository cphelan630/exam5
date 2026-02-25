# ReservingEngine Function Reference

All public and private functions defined in `src/reservingengine/reserving/`.

**Value column key:**
- `Genuine` — implements domain logic not available in chainladder
- `Orchestration` — thin wrapper, but provides uniform interface across methods

Functions previously listed as `Thin wrapper` have been removed — use chainladder's API directly.

---

## `triangle_io.py` — Triangle IO Utilities

| Function | Value | Description |
|---|---|---|
| `_import_chainladder` | Orchestration | Lazy-import guard for the `chainladder` package; raises a clear error if not installed. |
| `project_root` | Orchestration | Return the repository root based on this module's location. |
| `default_chainladder_data_dir` | Orchestration | Return the default local chainladder sample-data directory. |
| `resolve_dataset_path` | Orchestration | Resolve a dataset name to a CSV path in the local chainladder data folder. |
| `load_raw_dataset` | Orchestration | Load a local chainladder sample CSV as a `pd.DataFrame`. |
| `_triangle_to_frame` | Orchestration | **Private.** `triangle.to_frame(keepdims=True)` with a try/except API-compatibility fallback; used by all internal functions. |
| `validate_triangle_totals_match_raw` | Genuine | Validates that raw tabular column totals equal the Triangle total; data integrity check. |
| `validate_cum_incr_roundtrip` | Genuine | Tests cumulative → incremental → cumulative round-trip consistency with tolerance. |
| `_align_exposure_vector` | Genuine | **Private.** Fuzzy-matches an exposure `pd.Series` to triangle origin labels (tries str fallback). |
| `build_exposure_triangle` | Genuine | Constructs a sample-weight Triangle aligned to another Triangle's latest diagonal. |

**Chainladder equivalents for removed functions:**
- `build_triangle(df, origin_col=..., value_cols=..., cumulative=...)` → `cl.Triangle(df, origin=..., columns=..., cumulative=...)`
- `triangle_to_frame(tri)` → `tri.to_frame(origin_as_datetime=False, keepdims=True)`
- `triangle_total(tri)` → `tri.to_frame().select_dtypes('number').values.sum()`

---

## `methods.py` — Reserving Methods

### Dataclass

| Class | Value | Description |
|---|---|---|
| `MethodResult` | Genuine | Compact summary for one reserving method run (totals + assumption notes). Standardises output across all methods. |

### Internal Helpers

| Function | Value | Description |
|---|---|---|
| `_detect_value_column` | Genuine | Detect the numeric value column from a Triangle-converted DataFrame (multi-column safe). |
| `_segment_key_from_row` | Genuine | Build a string segment key from a DataFrame row and segment column list. |
| `_triangle_series_by_origin` | Genuine | Extract a `pd.Series` keyed by `(segment, origin)` from a Triangle. |
| `_normalize_summary_index` | Genuine | Flatten a single-segment MultiIndex summary to a plain origin index. |
| `_pattern_dict` | Genuine | Convert a pattern Triangle (LDF/CDF) to a nested `dict[segment][age]` dict. |
| `_summary_from_model` | Genuine | Build a `latest / ultimate / ibnr` DataFrame from a fitted chainladder model. |
| `_method_result` | Genuine | Construct a `MethodResult` from a summary DataFrame. |
| `_fit_development` | Orchestration | **Private.** `cl.Development(...).fit_transform(tri)` — shared by all Track B methods. |
| `_apply_tail` | Orchestration | **Private.** `cl.TailConstant/TailCurve(...).fit_transform(tri)`. |
| `_prepare_track_a_triangle` | Orchestration | Apply `_fit_development` + optional tail; return prepared Triangle and model artifacts. |
| `_run_track_a_estimator` | Genuine | Core orchestration: development prep → estimator fit → artifact extraction. Shared by all Track A methods. |
| `_triangle_to_segment_wide` | Genuine | Split a Triangle into `dict[segment → wide DataFrame]` for Track B methods. |
| `_align_external_origin_series` | Genuine | Align an external Series to a Triangle summary index (plain or MultiIndex, with str fallback). |
| `_origin_to_year_series` | Genuine | Convert an origin index to numeric calendar years for log-linear trend fitting. |

### Public Track A Functions

| Function | Value | Description |
|---|---|---|
| `run_chain_ladder` | Orchestration | Track A Chain Ladder; returns a uniform `(MethodResult, summary, artifacts)` triple. |
| `run_expected_loss` | Orchestration | Track A Expected Loss with exposure weighting; uniform return interface. |
| `run_bornhuetter_ferguson` | Orchestration | Track A Bornhuetter-Ferguson with exposure; uniform return interface. |
| `run_benktander` | Orchestration | Track A Benktander with exposure; uniform return interface. |
| `run_cape_cod` | Orchestration | Track A Cape Cod with exposure, trend, decay; uniform return interface. |
| `run_case_outstanding_chainladder` | Orchestration | Uses `cl.CaseOutstanding` + `Chainladder`; joins two triangles and standardises output. |

**Chainladder equivalents for removed functions:**
- `fit_development(tri, ...)` → `cl.Development(...).fit_transform(tri)`
- `apply_tail(tri, tail_kind='constant', ...)` → `cl.TailConstant(**kwargs).fit_transform(tri)` or `cl.TailCurve(...)`
- `compare_method_outputs(results)` → `pd.DataFrame([vars(r) for r in results])`

### Public Track B Functions

| Function | Value | Description |
|---|---|---|
| `run_case_outstanding_friedland` | **Genuine** | Friedland Ch. 12: paid-to-prior-case + case-to-prior-case ratio projection. No chainladder equivalent. |
| `run_frequency_severity_friedland` | **Genuine** | Friedland Ch. 11 Technique 1: develop counts and severities separately, combine. No chainladder equivalent. |
| `run_frequency_severity_technique2` | **Genuine** | Friedland Ch. 11 Technique 2: log-linear trend projection from mature years applied to exposure. Entirely custom. |
| `run_disposal_rate` | **Genuine** | Friedland Ch. 13: disposal rate method with monotone enforcement and incremental severity. Entirely custom. |

---

## `diagnostics.py` — Triangle Diagnostics

### Internal Helpers

| Function | Value | Description |
|---|---|---|
| `_triangle_to_wide` | Genuine | Convert a single-column Triangle to a wide DataFrame (origins × dev ages) — needed by multiple diagnostics. |
| `_medial_average` | Genuine | Drop the highest and lowest value, then simple-average the rest (Friedland medial). |
| `_origin_year_numeric` | Genuine | Parse an origin label string to a calendar-year integer. |
| `_is_number` | Orchestration | Internal: `float(s)` in a try/except; used only by `trend_summary`. |

### Public Diagnostic Functions

| Function | Value | Description |
|---|---|---|
| `link_ratio_table` | **Genuine** | Friedland Ch. 8 Exhibit 1: per-origin link ratios + Vol Wtd / Simple / Medial / Selected LDF rows. chainladder has `.link_ratio` but no formatted exhibit. |
| `plot_link_ratio_heatmap` | **Genuine** | Seaborn heatmap of link ratios, centered at column median. chainladder's `.heatmap()` shows values, not ratios. |
| `calendar_year_diagnostic` | **Genuine** | Friedland L/S test: label each factor L/S vs. column median, map to calendar year, count by year. Not in chainladder. |
| `paid_vs_incurred_comparison` | **Genuine** | Side-by-side LDF, CDF, and latest-diagonal paid-to-incurred ratios across two triangles. |
| `trend_summary` | **Genuine** | Slope-based trend analysis on link ratios by age, origin, and calendar-year diagonal. No chainladder equivalent. |

**Chainladder equivalents for removed functions:**
- `triangle_sanity_checks(tri)` → `tri.to_frame().describe()`
- `selected_ldf_table(tri, **kwargs)` → `cl.Development(**kwargs).fit(tri).ldf_.to_frame()`
- `percent_paid_reported(tri)` → fit `cl.Chainladder()` then access `.ultimate_`, `.ibnr_`, `.latest_diagonal`
- `projection_comparison(paid, reported)` → fit two `cl.Chainladder()` models and join results
- `count_triangle_diagnostics(tri)` → `link_ratio_table(tri)` + `1 - 1 / lr.loc["Vol Wtd Avg"]`
- `case_adequacy_indicator(paid, incurred)` → `paid_vs_incurred_comparison(paid, incurred)` + `np.polyfit`
- `run_correlation_tests(tri)` → `cl.DevelopmentCorrelation(tri)` and `cl.ValuationCorrelation(tri)`

---

## `compare.py` — Baseline Snapshots & Reconciliation

### Internal Helpers

| Function | Value | Description |
|---|---|---|
| `_to_float_or_none` | Orchestration | `pd.to_numeric(v, errors='coerce')` with NaN→None conversion. |
| `_summary_to_nested_dict` | Genuine | Serialises a summary DataFrame column to a nested `dict[segment][origin]` structure. |
| `_iter_nested_metric_values` | Genuine | Yields `(segment, origin, value)` from a nested metrics dict. |
| `_track_key` | Orchestration | String concatenation for tolerance dict keys. |
| `_reason_code` | Genuine | Returns a structured failure reason code based on method name and tail assumptions. |
| `_flatten_pattern` | Genuine | Recursively flattens a nested pattern dict to `("|"-joined key, value)` pairs. |

### Public Functions

| Function | Value | Description |
|---|---|---|
| `snapshot_method_output` | **Genuine** | Serialises a full method run (result + artifacts + metadata) to a JSON-compatible dict for baseline storage. |
| `compare_results_to_baseline` | **Genuine** | Differential testing of totals and AY-level values against a snapshot, with per-track tolerances and reason codes. |
| `compare_patterns_to_baseline` | **Genuine** | Pattern-level (LDF/CDF) regression comparison against a snapshot with absolute tolerance. |
| `build_reconciliation_report` | **Genuine** | Combines result + pattern comparisons into a single report sorted by pass/fail. |

**Chainladder equivalents for removed functions:**
- `load_baseline_fixture(path)` → `json.loads(Path(path).read_text())`
- `save_baseline_fixture(path, payload)` → `Path(path).write_text(json.dumps(payload, indent=2))`

---

## Summary

| Module | Genuine | Orchestration | Removed (thin wrappers) |
|---|---|---|---|
| `triangle_io.py` | 3 | 7 | 3 (`build_triangle`, `triangle_to_frame`→private, `triangle_total`) |
| `methods.py` | 11 | 8 | 3 (`fit_development`→private, `apply_tail`→private, `compare_method_outputs`) |
| `diagnostics.py` | 7 | 1 | 7 (`triangle_sanity_checks`, `selected_ldf_table`, `percent_paid_reported`, `projection_comparison`, `count_triangle_diagnostics`, `case_adequacy_indicator`, `run_correlation_tests`) |
| `compare.py` | 6 | 2 | 2 (`load_baseline_fixture`, `save_baseline_fixture`) |
| **Total public** | **27** | **18** | **15 removed** |
