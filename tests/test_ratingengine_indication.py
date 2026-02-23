import pytest

from ratingengine.indication import (
    explain_equivalence,
    loss_ratio_indication,
    pure_premium_indication,
)


def test_pure_premium_indication_numeric() -> None:
    out = pure_premium_indication(
        losses=100.0,
        lae=20.0,
        exposures=10.0,
        fixed_exp_per_exp=3.0,
        var_exp_ratio=0.20,
        profit_ratio=0.05,
        current_premium_per_exposure=18.0,
    )
    assert out.loss_plus_lae_per_exposure == pytest.approx(12.0)
    assert out.indicated_premium_per_exposure == pytest.approx(20.0)
    assert out.indicated_change == pytest.approx((20.0 / 18.0) - 1.0)


def test_loss_ratio_indication_numeric() -> None:
    out = loss_ratio_indication(
        ultimate_losses=100.0,
        lae=20.0,
        earned_prem_onlevel=200.0,
        fixed_exp_ratio=0.10,
        var_exp_ratio=0.20,
        profit_ratio=0.05,
    )
    assert out.actual_loss_ratio == pytest.approx(0.60)
    assert out.permissible_loss_ratio == pytest.approx(0.65)
    assert out.indicated_change == pytest.approx((0.60 / 0.65) - 1.0)


def test_indication_validation() -> None:
    with pytest.raises(ValueError):
        pure_premium_indication(
            losses=10.0,
            lae=0.0,
            exposures=0.0,
            fixed_exp_per_exp=1.0,
            var_exp_ratio=0.2,
            profit_ratio=0.1,
        )
    with pytest.raises(ValueError):
        pure_premium_indication(
            losses=10.0,
            lae=0.0,
            exposures=10.0,
            fixed_exp_per_exp=1.0,
            var_exp_ratio=0.7,
            profit_ratio=0.3,
        )
    with pytest.raises(ValueError):
        loss_ratio_indication(
            ultimate_losses=10.0,
            lae=0.0,
            earned_prem_onlevel=100.0,
            fixed_exp_ratio=0.5,
            var_exp_ratio=0.4,
            profit_ratio=0.2,
        )


def test_explain_equivalence_contains_expected_anchors() -> None:
    text = explain_equivalence()
    assert "Pure premium form" in text
    assert "Loss ratio form" in text
    assert "same indicated premium level" in text
