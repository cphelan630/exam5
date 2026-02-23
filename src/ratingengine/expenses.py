"""Expense provision utilities for pricing indications."""

from __future__ import annotations

import math

from .types import ExpenseProvisionResult


def _validate_positive(value: float, *, name: str) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be a finite value greater than 0.")
    return val


def _validate_ratio(value: float, *, name: str) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite.")
    return val


def permissible_loss_ratio(
    variable_expense_ratio: float,
    fixed_expense_ratio: float,
    profit_ratio: float,
) -> float:
    """Compute permissible loss ratio (PLR)."""

    var = _validate_ratio(variable_expense_ratio, name="variable_expense_ratio")
    fixed = _validate_ratio(fixed_expense_ratio, name="fixed_expense_ratio")
    profit = _validate_ratio(profit_ratio, name="profit_ratio")
    plr = 1.0 - var - fixed - profit
    if plr <= 1e-12:
        raise ValueError(
            "variable_expense_ratio + fixed_expense_ratio + profit_ratio < 1 is required."
        )
    return plr


def expense_provisions_all_variable(
    variable_expenses: float,
    fixed_expenses: float,
    premium: float,
    *,
    profit_ratio: float = 0.0,
) -> ExpenseProvisionResult:
    """Treat all expenses as premium-variable."""

    premium_val = _validate_positive(premium, name="premium")
    variable_ratio = (float(variable_expenses) + float(fixed_expenses)) / premium_val
    profit = _validate_ratio(profit_ratio, name="profit_ratio")
    plr = permissible_loss_ratio(variable_ratio, 0.0, profit)
    return ExpenseProvisionResult(
        method="all_variable",
        variable_expense_ratio=variable_ratio,
        fixed_expense_ratio=0.0,
        fixed_expense_per_exposure=None,
        profit_ratio=profit,
        permissible_loss_ratio=plr,
    )


def expense_provisions_premium_based(
    variable_expenses: float,
    fixed_expenses: float,
    premium: float,
    projected_premium: float,
    *,
    profit_ratio: float = 0.0,
) -> ExpenseProvisionResult:
    """Allocate fixed expenses as a premium ratio on projected premium."""

    premium_val = _validate_positive(premium, name="premium")
    projected_val = _validate_positive(projected_premium, name="projected_premium")
    variable_ratio = float(variable_expenses) / premium_val
    fixed_ratio = float(fixed_expenses) / projected_val
    profit = _validate_ratio(profit_ratio, name="profit_ratio")
    plr = permissible_loss_ratio(variable_ratio, fixed_ratio, profit)
    return ExpenseProvisionResult(
        method="premium_based",
        variable_expense_ratio=variable_ratio,
        fixed_expense_ratio=fixed_ratio,
        fixed_expense_per_exposure=None,
        profit_ratio=profit,
        permissible_loss_ratio=plr,
    )


def expense_provisions_exposure_based(
    variable_expenses: float,
    fixed_expenses: float,
    premium: float,
    exposures: float,
    *,
    profit_ratio: float = 0.0,
) -> ExpenseProvisionResult:
    """Allocate fixed expenses as a per-exposure amount."""

    premium_val = _validate_positive(premium, name="premium")
    exposures_val = _validate_positive(exposures, name="exposures")
    variable_ratio = float(variable_expenses) / premium_val
    fixed_per_exposure = float(fixed_expenses) / exposures_val
    profit = _validate_ratio(profit_ratio, name="profit_ratio")
    plr = permissible_loss_ratio(variable_ratio, 0.0, profit)
    return ExpenseProvisionResult(
        method="exposure_based",
        variable_expense_ratio=variable_ratio,
        fixed_expense_ratio=0.0,
        fixed_expense_per_exposure=fixed_per_exposure,
        profit_ratio=profit,
        permissible_loss_ratio=plr,
    )
