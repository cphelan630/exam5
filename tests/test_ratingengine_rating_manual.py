import pandas as pd
import pytest

from ratingengine.rating_manual import rate_book, rate_policy


def test_rate_policy_breakdown() -> None:
    out = rate_policy(
        exposure=10.0,
        base_rate=100.0,
        relativity_factors=[1.2, 0.8],
        modifier_factors=[1.1],
        fixed_fee=25.0,
        min_premium=1100.0,
        round_to=50.0,
    )
    assert out.relativity_product == pytest.approx(0.96)
    assert out.modifier_product == pytest.approx(1.1)
    assert out.variable_premium == pytest.approx(1056.0)
    assert out.premium_before_min == pytest.approx(1081.0)
    assert out.premium_after_min == pytest.approx(1100.0)
    assert out.premium_final == pytest.approx(1100.0)


def test_rate_book_batch_and_missing_lookup() -> None:
    df = pd.DataFrame(
        {
            "policy_id": [1, 2],
            "exposure": [1.0, 2.0],
            "territory": ["urban", "rural"],
            "limit": ["high", "low"],
            "modifier": [1.0, 0.9],
        }
    )
    tables = {
        "territory": {"urban": 1.2, "rural": 0.8},
        "limit": {"high": 1.5, "low": 1.0},
    }
    rated = rate_book(
        df,
        exposure_col="exposure",
        base_rate=100.0,
        relativity_tables=tables,
        fixed_fee=10.0,
        modifier_cols=["modifier"],
    )
    assert rated.loc[0, "premium_final"] == pytest.approx(190.0)
    assert rated.loc[1, "premium_final"] == pytest.approx(154.0)

    bad_tables = {"territory": {"urban": 1.2}}
    with pytest.raises(KeyError):
        rate_book(
            df,
            exposure_col="exposure",
            base_rate=100.0,
            relativity_tables=bad_tables,
        )
