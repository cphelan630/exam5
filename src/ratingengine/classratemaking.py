"""Univariate class ratemaking helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"DataFrame is missing required columns: {joined}")


def univariate_pp_rels(
    df: pd.DataFrame,
    class_var: str,
    loss_col: str,
    exposure_col: str,
) -> pd.DataFrame:
    """Compute univariate pure premium relativities by class."""

    _require_columns(df, [class_var, loss_col, exposure_col])
    grouped = (
        df.groupby(class_var, dropna=False, observed=False)[[loss_col, exposure_col]]
        .sum()
        .rename(columns={loss_col: "loss", exposure_col: "exposure"})
    )

    total_exposure = float(grouped["exposure"].sum())
    if total_exposure <= 0.0:
        raise ValueError("Total exposure must be greater than 0.")

    overall_pp = float(grouped["loss"].sum()) / total_exposure
    grouped["metric"] = np.where(
        grouped["exposure"] != 0.0,
        grouped["loss"] / grouped["exposure"],
        np.nan,
    )
    grouped["relativity"] = grouped["metric"] / overall_pp
    out = grouped.reset_index().rename(columns={class_var: "class_value"})
    return out.loc[:, ["class_value", "loss", "exposure", "metric", "relativity"]]


def univariate_lr_rels(
    df: pd.DataFrame,
    class_var: str,
    loss_col: str,
    prem_col: str,
) -> pd.DataFrame:
    """Compute univariate loss ratio relativities by class."""

    _require_columns(df, [class_var, loss_col, prem_col])
    grouped = (
        df.groupby(class_var, dropna=False, observed=False)[[loss_col, prem_col]]
        .sum()
        .rename(columns={loss_col: "loss", prem_col: "premium"})
    )

    total_premium = float(grouped["premium"].sum())
    if total_premium <= 0.0:
        raise ValueError("Total premium must be greater than 0.")

    overall_lr = float(grouped["loss"].sum()) / total_premium
    grouped["metric"] = np.where(
        grouped["premium"] != 0.0,
        grouped["loss"] / grouped["premium"],
        np.nan,
    )
    grouped["relativity"] = grouped["metric"] / overall_lr
    out = grouped.reset_index().rename(columns={class_var: "class_value"})
    return out.loc[:, ["class_value", "loss", "premium", "metric", "relativity"]]


def apply_rels(
    base_rate: float,
    rels: Mapping[object, float] | pd.Series | pd.DataFrame,
    fixed_fee: float = 0.0,
) -> pd.Series:
    """Apply class relativities to a base rate and return class-level premiums."""

    base = float(base_rate)
    fee = float(fixed_fee)
    if isinstance(rels, pd.DataFrame):
        if rels.shape[1] != 1:
            raise ValueError("rels DataFrame must contain exactly one column.")
        rel_series = rels.iloc[:, 0].astype(float)
    elif isinstance(rels, pd.Series):
        rel_series = rels.astype(float)
    else:
        rel_series = pd.Series(dict(rels), dtype=float)

    out = base * rel_series + fee
    out.name = "indicated_premium"
    return out
