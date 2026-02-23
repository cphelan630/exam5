import math

import pandas as pd
import pytest

from ratingengine.data_prep import add_time_columns, aggregate_experience, safe_divide


def test_aggregate_experience_and_zero_denominator_handling() -> None:
    df = pd.DataFrame(
        {
            "state": ["A", "B", "C"],
            "exposure": [10.0, 5.0, 0.0],
            "premium": [100.0, 50.0, 0.0],
            "loss": [60.0, 20.0, 10.0],
            "lae": [6.0, 2.0, 0.0],
        }
    )
    out = aggregate_experience(df, group_cols="state").set_index("state")
    assert out.loc["A", "pure_premium"] == pytest.approx(6.6)
    assert out.loc["A", "loss_ratio"] == pytest.approx(0.66)
    assert out.loc["B", "pure_premium"] == pytest.approx(4.4)
    assert out.loc["B", "loss_ratio"] == pytest.approx(0.44)
    assert math.isnan(float(out.loc["C", "pure_premium"]))
    assert math.isnan(float(out.loc["C", "loss_ratio"]))


def test_add_time_columns_and_safe_divide_scalar() -> None:
    df = pd.DataFrame({"policy_date": ["2024-01-15", "2024-12-31"]})
    out = add_time_columns(df, "policy_date", prefix="policy")
    assert out.loc[0, "policy_year"] == 2024
    assert out.loc[0, "policy_month"] == 1
    assert out.loc[1, "policy_quarter"] == 4

    assert safe_divide(10.0, 0.0) != safe_divide(10.0, 10.0)
    assert math.isnan(float(safe_divide(10.0, 0.0)))
