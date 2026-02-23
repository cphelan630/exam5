from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from ratingengine.classratemaking import apply_rels
from ratingengine.credibility import limited_fluctuation_z
from ratingengine.data_prep import aggregate_experience, safe_divide
from ratingengine.loss_adj import exclude_catastrophes
from ratingengine.onlevel import average_rate_level, parallelogram_weights
from ratingengine.rating_manual import rate_book, rate_policy
from ratingengine.trend import trend_to_average_date, two_step_trend
from ratingengine.types import (
    ExpenseProvisionResult,
    LossAdjustmentResult,
    LossRatioIndicationResult,
    PurePremiumIndicationResult,
    RatedPolicyResult,
    TrendResult,
)


def test_result_dataclasses_are_constructible_and_frozen() -> None:
    trend = TrendResult(
        base_level=100.0,
        years_retro=1.0,
        years_pros=0.5,
        retro_factor=1.03,
        pros_factor=1.02,
        total_factor=1.0506,
        trended_level=105.06,
    )
    pp = PurePremiumIndicationResult(
        losses=100.0,
        lae=20.0,
        exposures=10.0,
        loss_plus_lae=120.0,
        loss_plus_lae_per_exposure=12.0,
        fixed_exp_per_exp=2.0,
        var_exp_ratio=0.20,
        profit_ratio=0.05,
        denominator=0.75,
        indicated_premium_per_exposure=18.6666666667,
        current_premium_per_exposure=18.0,
        indicated_change=0.0370370370,
    )
    lr = LossRatioIndicationResult(
        ultimate_losses=100.0,
        lae=20.0,
        earned_premium_onlevel=200.0,
        actual_loss_ratio=0.60,
        fixed_exp_ratio=0.10,
        var_exp_ratio=0.20,
        profit_ratio=0.05,
        permissible_loss_ratio=0.65,
        indicated_change=-0.0769230769,
    )
    expense = ExpenseProvisionResult(
        method="premium_based",
        variable_expense_ratio=0.20,
        fixed_expense_ratio=0.05,
        fixed_expense_per_exposure=None,
        profit_ratio=0.05,
        permissible_loss_ratio=0.70,
    )
    loss_adj = LossAdjustmentResult(
        input_total=100.0,
        adjusted_total=105.0,
        adjustment_factor=1.05,
        components={"trend_factor": 1.05},
    )
    rated = RatedPolicyResult(
        base_rate=100.0,
        exposure=2.0,
        relativity_product=1.1,
        modifier_product=1.0,
        variable_premium=220.0,
        fixed_fee=15.0,
        premium_before_min=235.0,
        premium_after_min=235.0,
        premium_final=235.0,
    )

    assert trend.trended_level == pytest.approx(105.06)
    assert pp.denominator == pytest.approx(0.75)
    assert lr.permissible_loss_ratio == pytest.approx(0.65)
    assert expense.method == "premium_based"
    assert loss_adj.adjustment_factor == pytest.approx(1.05)
    assert rated.premium_final == pytest.approx(235.0)

    with pytest.raises(FrozenInstanceError):
        trend.base_level = 200.0  # type: ignore[misc]


def test_pandas_routines_series_alignment_and_aggregation() -> None:
    numer = pd.Series([10.0, 20.0], index=["A", "B"])
    denom = pd.Series([2.0], index=["A"])
    out = safe_divide(numer, denom)
    assert out["A"] == pytest.approx(5.0)
    assert pd.isna(out["B"])

    df = pd.DataFrame(
        {
            "exposure": [10.0, 5.0],
            "premium": [100.0, 55.0],
            "loss": [60.0, 33.0],
            "lae": [6.0, 3.0],
        }
    )
    agg = aggregate_experience(df)
    assert len(agg) == 1
    assert agg.loc[0, "pure_premium"] == pytest.approx((60.0 + 33.0 + 6.0 + 3.0) / 15.0)


def test_validation_branches_raise_expected_errors() -> None:
    with pytest.raises(KeyError):
        parallelogram_weights(
            [{"effective_date": "2024-01-01"}],
            ("2024-01-01", "2025-01-01"),
        )
    with pytest.raises(ValueError):
        average_rate_level([0.5], [1.0, 1.1])
    with pytest.raises(ValueError):
        trend_to_average_date(100.0, "2024-01-01", "2025-01-01", annual_rate=-1.0)
    with pytest.raises(ValueError):
        two_step_trend(
            100.0,
            "2024-01-01",
            "2025-01-01",
            0.02,
            0.03,
            as_of_date="2023-12-01",
        )
    with pytest.raises(ValueError):
        limited_fluctuation_z(10, 100, prob=1.0, error=0.05)
    with pytest.raises(ValueError):
        exclude_catastrophes(pd.Series([1.0, 2.0]), [True])
    with pytest.raises(ValueError):
        apply_rels(100.0, pd.DataFrame({"a": [1.0], "b": [1.1]}))
    with pytest.raises(ValueError):
        rate_policy(exposure=1.0, base_rate=100.0, relativity_factors=[1.0], round_to=0.0)
    with pytest.raises(KeyError):
        rate_book(
            pd.DataFrame({"x": [1.0]}),
            exposure_col="exposure",
            base_rate=100.0,
            relativity_tables={},
        )
