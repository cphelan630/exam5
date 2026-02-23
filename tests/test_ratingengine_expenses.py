import pytest

from ratingengine.expenses import (
    expense_provisions_all_variable,
    expense_provisions_exposure_based,
    expense_provisions_premium_based,
    permissible_loss_ratio,
)


def test_expense_provisions_all_variable() -> None:
    out = expense_provisions_all_variable(
        variable_expenses=20.0,
        fixed_expenses=10.0,
        premium=100.0,
        profit_ratio=0.05,
    )
    assert out.variable_expense_ratio == pytest.approx(0.30)
    assert out.fixed_expense_ratio == pytest.approx(0.0)
    assert out.permissible_loss_ratio == pytest.approx(0.65)


def test_expense_provisions_premium_and_exposure_methods() -> None:
    premium_based = expense_provisions_premium_based(
        variable_expenses=20.0,
        fixed_expenses=10.0,
        premium=100.0,
        projected_premium=200.0,
        profit_ratio=0.05,
    )
    assert premium_based.variable_expense_ratio == pytest.approx(0.20)
    assert premium_based.fixed_expense_ratio == pytest.approx(0.05)
    assert premium_based.permissible_loss_ratio == pytest.approx(0.70)

    exposure_based = expense_provisions_exposure_based(
        variable_expenses=20.0,
        fixed_expenses=10.0,
        premium=100.0,
        exposures=5.0,
        profit_ratio=0.05,
    )
    assert exposure_based.variable_expense_ratio == pytest.approx(0.20)
    assert exposure_based.fixed_expense_per_exposure == pytest.approx(2.0)
    assert exposure_based.permissible_loss_ratio == pytest.approx(0.75)


def test_plr_validation() -> None:
    with pytest.raises(ValueError):
        permissible_loss_ratio(0.5, 0.4, 0.2)
