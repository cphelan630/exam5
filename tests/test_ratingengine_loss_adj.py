import pandas as pd
import pytest

from ratingengine.loss_adj import (
    apply_loss_adjustment_factors,
    cap_losses,
    exclude_catastrophes,
)


def test_cap_losses_and_exclude_catastrophes() -> None:
    losses = pd.Series([50.0, 200.0])
    capped = cap_losses(losses, cap=100.0)
    assert capped.tolist() == [50.0, 100.0]

    adjusted = exclude_catastrophes(capped, [False, True])
    assert adjusted.tolist() == [50.0, 0.0]


def test_apply_loss_adjustment_factors() -> None:
    result = apply_loss_adjustment_factors(
        100.0,
        development_factor=1.10,
        trend_factor=1.05,
        benefit_factor=0.98,
        limit_factor=1.02,
        onlevel_factor=1.01,
        cat_load_factor=1.03,
    )
    expected = 1.10 * 1.05 * 0.98 * 1.02 * 1.01 * 1.03
    assert result.adjustment_factor == pytest.approx(expected)
    assert result.adjusted_total == pytest.approx(100.0 * expected)

    with pytest.raises(ValueError):
        apply_loss_adjustment_factors(100.0, trend_factor=0.0)
