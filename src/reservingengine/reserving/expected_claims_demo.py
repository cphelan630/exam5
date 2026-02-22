"""Expected-claims (ECR/ELR) helpers for teaching/demo notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_claims_from_premium(
    earned_premium: pd.Series,
    expected_claim_ratio: float | pd.Series,
) -> pd.Series:
    """Ultimate claims estimate from earned premium x expected claim ratio."""

    ecr = _to_series(expected_claim_ratio, earned_premium.index)
    return earned_premium.astype(float) * ecr


def expected_claims_from_exposure(
    earned_exposure: pd.Series,
    expected_pure_premium: float | pd.Series,
) -> pd.Series:
    """Ultimate claims estimate from earned exposures x expected pure premium."""

    epp = _to_series(expected_pure_premium, earned_exposure.index)
    return earned_exposure.astype(float) * epp


def implied_ecr(
    claims: pd.Series,
    earned_premium: pd.Series,
) -> pd.Series:
    """Claim ratio by AY with safe divide handling."""

    premium = earned_premium.astype(float)
    out = claims.astype(float) / premium
    return out.where(premium != 0, np.nan)


def selected_ecr_from_history(
    historical_claims: pd.Series,
    historical_premium: pd.Series,
    *,
    method: str = "volume_weighted",
) -> float:
    """Select ECR from historical ratios."""

    ratios = implied_ecr(historical_claims, historical_premium).dropna()
    if method == "arithmetic":
        return float(ratios.mean())
    if method == "median":
        return float(ratios.median())
    if method == "volume_weighted":
        denom = historical_premium.astype(float).sum()
        if denom == 0:
            return np.nan
        return float(historical_claims.astype(float).sum() / denom)
    raise ValueError("method must be one of: arithmetic, median, volume_weighted.")


def adjust_selected_ecr(
    base_ecr: float,
    *,
    level_adjustment: float = 1.0,
) -> float:
    """Apply a multiplicative level adjustment to selected ECR."""

    return float(base_ecr) * float(level_adjustment)


def ibnr_from_expected_claims(
    expected_ultimate: pd.Series,
    paid_to_date: pd.Series,
) -> pd.Series:
    """IBNR estimate from expected ultimate minus paid-to-date."""

    return expected_ultimate.astype(float) - paid_to_date.astype(float)


def build_environment_impact_table() -> pd.DataFrame:
    """Impacts on expected-claims method (paid vs reported) for common changes."""

    return pd.DataFrame(
        [
            [
                "Increase in exposure",
                "No material effect if average accident date is unchanged",
                "No material effect if average accident date is unchanged",
            ],
            [
                "Average accident date shifts forward",
                "Underestimates ultimate (usually less than development method)",
                "Underestimates ultimate (usually less than development method)",
            ],
            [
                "Increase claim ratios",
                "If not reflected in selected ECR, ultimates are underestimated",
                "If not reflected in selected ECR, ultimates are underestimated",
            ],
            [
                "Speedup in claim settlement rate",
                "Overestimates ultimate (usually less than development method)",
                "No material effect",
            ],
            [
                "Increase in case outstanding adequacy",
                "No material effect",
                "Overestimates ultimate (usually less than development method)",
            ],
            [
                "Change in product mix",
                "Impacted when segments have different ECRs/development patterns",
                "Impacted when segments have different ECRs/development patterns",
            ],
        ],
        columns=["Description", "Paid impact", "Reported impact"],
    )


def _to_series(value: float | pd.Series, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index).astype(float)
    return pd.Series(float(value), index=index, dtype=float)
