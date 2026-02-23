"""Rating-manual style helpers for base rate and relativity pricing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from .types import RatedPolicyResult


def _factor_product(factors: Mapping[object, float] | Sequence[float] | None) -> float:
    if factors is None:
        return 1.0

    if isinstance(factors, Mapping):
        values = factors.values()
    else:
        values = factors

    product = 1.0
    for factor in values:
        value = float(factor)
        if value <= 0.0:
            raise ValueError("All factors must be positive.")
        product *= value
    return product


def rate_policy(
    exposure: float,
    base_rate: float,
    relativity_factors: Mapping[object, float] | Sequence[float],
    *,
    modifier_factors: Mapping[object, float] | Sequence[float] | None = None,
    fixed_fee: float = 0.0,
    min_premium: float | None = None,
    round_to: float | None = None,
) -> RatedPolicyResult:
    """Rate a single policy using base rate, relativities, and modifiers."""

    exposure_val = float(exposure)
    base_rate_val = float(base_rate)
    fee = float(fixed_fee)

    if exposure_val < 0.0:
        raise ValueError("exposure must be non-negative.")
    if base_rate_val < 0.0:
        raise ValueError("base_rate must be non-negative.")
    if fee < 0.0:
        raise ValueError("fixed_fee must be non-negative.")

    rel_prod = _factor_product(relativity_factors)
    mod_prod = _factor_product(modifier_factors)
    variable_premium = base_rate_val * exposure_val * rel_prod * mod_prod
    premium_before_min = variable_premium + fee

    premium_after_min = premium_before_min
    if min_premium is not None:
        min_val = float(min_premium)
        if min_val < 0.0:
            raise ValueError("min_premium must be non-negative.")
        premium_after_min = max(premium_after_min, min_val)

    premium_final = premium_after_min
    if round_to is not None:
        round_step = float(round_to)
        if round_step <= 0.0:
            raise ValueError("round_to must be greater than 0.")
        premium_final = round(premium_after_min / round_step) * round_step

    return RatedPolicyResult(
        base_rate=base_rate_val,
        exposure=exposure_val,
        relativity_product=rel_prod,
        modifier_product=mod_prod,
        variable_premium=variable_premium,
        fixed_fee=fee,
        premium_before_min=premium_before_min,
        premium_after_min=premium_after_min,
        premium_final=premium_final,
    )


def rate_book(
    df: pd.DataFrame,
    *,
    exposure_col: str,
    base_rate: float,
    relativity_tables: Mapping[str, Mapping[object, float]],
    fixed_fee: float = 0.0,
    modifier_cols: Sequence[str] | None = None,
    min_premium: float | None = None,
    round_to: float | None = None,
) -> pd.DataFrame:
    """Batch-rate a DataFrame of policies."""

    if exposure_col not in df.columns:
        raise KeyError(f"DataFrame is missing exposure column: {exposure_col}")

    out = df.copy()
    factor_cols: list[str] = []

    for col_name, table in relativity_tables.items():
        if col_name not in out.columns:
            raise KeyError(f"DataFrame is missing relativity column: {col_name}")
        factor_col = f"{col_name}_factor"
        out[factor_col] = out[col_name].map(dict(table))
        missing_mask = out[factor_col].isna()
        if missing_mask.any():
            missing_values = out.loc[missing_mask, col_name].drop_duplicates().tolist()
            raise KeyError(
                f"Missing relativity lookup values for {col_name}: {missing_values}"
            )
        factor_cols.append(factor_col)

    modifiers = list(modifier_cols or [])
    for mod_col in modifiers:
        if mod_col not in out.columns:
            raise KeyError(f"DataFrame is missing modifier column: {mod_col}")

    rows: list[dict[str, float]] = []
    for _, row in out.iterrows():
        rel_factors = [float(row[col]) for col in factor_cols]
        mod_factors = [float(row[col]) for col in modifiers] if modifiers else None
        rated = rate_policy(
            exposure=float(row[exposure_col]),
            base_rate=base_rate,
            relativity_factors=rel_factors,
            modifier_factors=mod_factors,
            fixed_fee=fixed_fee,
            min_premium=min_premium,
            round_to=round_to,
        )
        rows.append(
            {
                "relativity_product": rated.relativity_product,
                "modifier_product": rated.modifier_product,
                "variable_premium": rated.variable_premium,
                "premium_before_min": rated.premium_before_min,
                "premium_after_min": rated.premium_after_min,
                "premium_final": rated.premium_final,
            }
        )

    results = pd.DataFrame(rows, index=out.index)
    return pd.concat([out, results], axis=1)
