"""Baseline snapshots and reconciliation helpers for reserving methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .methods import MethodResult

DEFAULT_TOLERANCES = {
    "pattern_abs": 1e-8,
    "ay_rel_track_a": 1e-6,
    "ay_rel_track_b": 1e-4,
    "total_rel_track_a": 1e-8,
    "total_rel_track_b": 1e-5,
}


def _to_float_or_none(value: Any) -> float | None:
    val = pd.to_numeric(value, errors="coerce")
    if pd.isna(val):
        return None
    return float(val)


def _summary_to_nested_dict(
    summary: pd.DataFrame,
    column: str,
) -> dict[str, dict[str, float | None]]:
    if column not in summary.columns:
        return {}
    out: dict[str, dict[str, float | None]] = {}
    if isinstance(summary.index, pd.MultiIndex):
        for (segment_key, origin), value in summary[column].items():
            seg = str(segment_key)
            out.setdefault(seg, {})[str(origin)] = _to_float_or_none(value)
    else:
        seg = "(All)"
        out.setdefault(seg, {})
        for origin, value in summary[column].items():
            out[seg][str(origin)] = _to_float_or_none(value)
    return out


def snapshot_method_output(
    result: MethodResult,
    summary: pd.DataFrame,
    artifacts: Mapping[str, Any] | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a serializable baseline snapshot payload for one method."""

    artifacts = dict(artifacts or {})
    if isinstance(summary.index, pd.MultiIndex):
        segment_keys = [str(v) for v in summary.index.get_level_values(0).unique().tolist()]
    else:
        segment_keys = ["(All)"]

    valuation_date: str | None = None
    for key in ("model", "prepared_triangle"):
        obj = artifacts.get(key)
        if obj is None or not hasattr(obj, "valuation_date"):
            continue
        try:
            valuation_date = str(obj.valuation_date)
        except Exception:  # pragma: no cover - defensive conversion
            valuation_date = None
        if valuation_date is not None:
            break

    method_metadata = {
        "track": artifacts.get("track"),
        "tail_assumption": artifacts.get("tail_assumption", "none"),
        "segment_keys": segment_keys,
        "valuation_date": valuation_date,
    }
    method_metadata.update(dict(metadata or {}))
    return {
        "method_name": result.method_name,
        "selected_ldfs_by_age": artifacts.get("selected_ldfs_by_age", {}),
        "selected_cdfs_by_age": artifacts.get("selected_cdfs_by_age", {}),
        "latest_diagonal_by_ay": _summary_to_nested_dict(summary, "latest"),
        "ultimate_by_ay": _summary_to_nested_dict(summary, "ultimate"),
        "ibnr_by_ay": _summary_to_nested_dict(summary, "ibnr"),
        "totals": {
            "latest": _to_float_or_none(result.latest_reported_total),
            "ultimate": _to_float_or_none(result.ultimate_total),
            "ibnr": _to_float_or_none(result.ibnr_total),
        },
        "metadata": method_metadata,
    }


def load_baseline_fixture(path: str | Path) -> dict[str, Any]:
    """Load a JSON baseline fixture."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_baseline_fixture(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Save a JSON baseline fixture."""

    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _iter_nested_metric_values(
    nested: Mapping[str, Mapping[str, Any]],
) -> Iterable[tuple[str, str, float | None]]:
    for segment_key, by_origin in nested.items():
        for origin, value in by_origin.items():
            yield str(segment_key), str(origin), _to_float_or_none(value)


def _track_key(track: str, metric_scope: str) -> str:
    return f"{metric_scope}_rel_{track}"


def _reason_code(
    *,
    passed: bool,
    metric: str,
    has_missing: bool,
    baseline_tail: str | None,
    current_tail: str | None,
    method_name: str,
) -> str:
    if passed:
        return "OK"
    if metric.startswith("selected_"):
        return "PATTERN_MISMATCH"
    if baseline_tail != current_tail:
        return "TAIL_ASSUMPTION_DIFF"
    method_name_l = method_name.lower()
    if (
        "bornhuetter" in method_name_l
        or "cape cod" in method_name_l
        or "benktander" in method_name_l
    ):
        return "EXPOSURE_ALIGNMENT_DIFF"
    if has_missing:
        return "MISSINGNESS_DIFF"
    return "PATTERN_MISMATCH"


def compare_results_to_baseline(
    current_methods: Mapping[str, Mapping[str, Any]],
    baseline_methods: Mapping[str, Mapping[str, Any]],
    *,
    tolerances: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compare totals and AY-level latest/ultimate/IBNR values to baseline."""

    tol = dict(DEFAULT_TOLERANCES)
    tol.update(dict(tolerances or {}))
    rows: list[dict[str, Any]] = []

    for method_name, current in current_methods.items():
        baseline = baseline_methods.get(method_name, {})
        track = str(current.get("metadata", {}).get("track", "track_a"))
        baseline_tail = baseline.get("metadata", {}).get("tail_assumption")
        current_tail = current.get("metadata", {}).get("tail_assumption")

        total_rel_tol = float(tol.get(_track_key(track, "total"), 1e-6))
        ay_rel_tol = float(tol.get(_track_key(track, "ay"), 1e-6))

        for total_name in ("latest", "ultimate", "ibnr"):
            cur_val = _to_float_or_none(current.get("totals", {}).get(total_name))
            base_val = _to_float_or_none(baseline.get("totals", {}).get(total_name))
            has_missing = cur_val is None or base_val is None
            if has_missing:
                passed = False
                abs_diff = np.nan
                rel_diff = np.nan
            else:
                abs_diff = abs(cur_val - base_val)
                denom = abs(base_val) if abs(base_val) > 0 else 1.0
                rel_diff = abs_diff / denom
                passed = bool(rel_diff <= total_rel_tol)
            rows.append(
                {
                    "method_name": method_name,
                    "metric_group": "totals",
                    "metric_name": total_name,
                    "segment_key": None,
                    "origin": None,
                    "baseline_value": base_val,
                    "current_value": cur_val,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "tolerance": total_rel_tol,
                    "passed": passed,
                    "reason_code": _reason_code(
                        passed=passed,
                        metric=total_name,
                        has_missing=has_missing,
                        baseline_tail=baseline_tail,
                        current_tail=current_tail,
                        method_name=method_name,
                    ),
                }
            )

        for metric_name in ("latest_diagonal_by_ay", "ultimate_by_ay", "ibnr_by_ay"):
            cur_nested = current.get(metric_name, {})
            base_nested = baseline.get(metric_name, {})
            cur_map = {(seg, org): val for seg, org, val in _iter_nested_metric_values(cur_nested)}
            base_map = {
                (seg, org): val
                for seg, org, val in _iter_nested_metric_values(base_nested)
            }
            keys = sorted(set(cur_map.keys()).union(base_map.keys()))
            for segment_key, origin in keys:
                cur_val = _to_float_or_none(cur_map.get((segment_key, origin)))
                base_val = _to_float_or_none(base_map.get((segment_key, origin)))
                has_missing = cur_val is None or base_val is None
                if has_missing:
                    passed = False
                    abs_diff = np.nan
                    rel_diff = np.nan
                else:
                    abs_diff = abs(cur_val - base_val)
                    denom = abs(base_val) if abs(base_val) > 0 else 1.0
                    rel_diff = abs_diff / denom
                    passed = bool(rel_diff <= ay_rel_tol)
                rows.append(
                    {
                        "method_name": method_name,
                        "metric_group": metric_name,
                        "metric_name": metric_name,
                        "segment_key": segment_key,
                        "origin": origin,
                        "baseline_value": base_val,
                        "current_value": cur_val,
                        "abs_diff": abs_diff,
                        "rel_diff": rel_diff,
                        "tolerance": ay_rel_tol,
                        "passed": passed,
                        "reason_code": _reason_code(
                            passed=passed,
                            metric=metric_name,
                            has_missing=has_missing,
                            baseline_tail=baseline_tail,
                            current_tail=current_tail,
                            method_name=method_name,
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _flatten_pattern(
    obj: Any,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, float | None]]:
    if isinstance(obj, Mapping):
        rows: list[tuple[str, float | None]] = []
        for key, value in obj.items():
            rows.extend(_flatten_pattern(value, prefix + (str(key),)))
        return rows
    return [("|".join(prefix), _to_float_or_none(obj))]


def compare_patterns_to_baseline(
    current_methods: Mapping[str, Mapping[str, Any]],
    baseline_methods: Mapping[str, Mapping[str, Any]],
    *,
    tolerances: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compare selected LDF/CDF pattern snapshots to baseline."""

    tol = dict(DEFAULT_TOLERANCES)
    tol.update(dict(tolerances or {}))
    abs_tol = float(tol.get("pattern_abs", 1e-8))
    rows: list[dict[str, Any]] = []

    for method_name, current in current_methods.items():
        baseline = baseline_methods.get(method_name, {})
        baseline_tail = baseline.get("metadata", {}).get("tail_assumption")
        current_tail = current.get("metadata", {}).get("tail_assumption")
        for metric_name in ("selected_ldfs_by_age", "selected_cdfs_by_age"):
            cur_flat = dict(_flatten_pattern(current.get(metric_name, {})))
            base_flat = dict(_flatten_pattern(baseline.get(metric_name, {})))
            keys = sorted(set(cur_flat.keys()).union(base_flat.keys()))
            for key in keys:
                cur_val = _to_float_or_none(cur_flat.get(key))
                base_val = _to_float_or_none(base_flat.get(key))
                has_missing = cur_val is None or base_val is None
                if has_missing:
                    passed = False
                    abs_diff = np.nan
                else:
                    abs_diff = abs(cur_val - base_val)
                    passed = bool(abs_diff <= abs_tol)
                rows.append(
                    {
                        "method_name": method_name,
                        "metric_group": metric_name,
                        "metric_name": key,
                        "segment_key": None,
                        "origin": None,
                        "baseline_value": base_val,
                        "current_value": cur_val,
                        "abs_diff": abs_diff,
                        "rel_diff": np.nan,
                        "tolerance": abs_tol,
                        "passed": passed,
                        "reason_code": _reason_code(
                            passed=passed,
                            metric=metric_name,
                            has_missing=has_missing,
                            baseline_tail=baseline_tail,
                            current_tail=current_tail,
                            method_name=method_name,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_reconciliation_report(
    current_methods: Mapping[str, Mapping[str, Any]],
    baseline_methods: Mapping[str, Mapping[str, Any]],
    *,
    tolerances: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build a combined reconciliation report."""

    result_df = compare_results_to_baseline(
        current_methods,
        baseline_methods,
        tolerances=tolerances,
    )
    pattern_df = compare_patterns_to_baseline(
        current_methods,
        baseline_methods,
        tolerances=tolerances,
    )
    report = pd.concat([result_df, pattern_df], ignore_index=True)
    return report.sort_values(
        ["passed", "method_name", "metric_group"],
        ascending=[True, True, True],
    )
