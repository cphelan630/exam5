"""Shared result types for pricing and rating calculations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrendResult:
    """Result container for trend calculations."""

    base_level: float
    years_retro: float
    years_pros: float
    retro_factor: float
    pros_factor: float
    total_factor: float
    trended_level: float


@dataclass(frozen=True)
class PurePremiumIndicationResult:
    """Result container for pure premium indication."""

    losses: float
    lae: float
    exposures: float
    loss_plus_lae: float
    loss_plus_lae_per_exposure: float
    fixed_exp_per_exp: float
    var_exp_ratio: float
    profit_ratio: float
    denominator: float
    indicated_premium_per_exposure: float
    current_premium_per_exposure: float | None
    indicated_change: float | None


@dataclass(frozen=True)
class LossRatioIndicationResult:
    """Result container for loss ratio indication."""

    ultimate_losses: float
    lae: float
    earned_premium_onlevel: float
    actual_loss_ratio: float
    fixed_exp_ratio: float
    var_exp_ratio: float
    profit_ratio: float
    permissible_loss_ratio: float
    indicated_change: float


@dataclass(frozen=True)
class ExpenseProvisionResult:
    """Result container for expense provision outputs."""

    method: str
    variable_expense_ratio: float
    fixed_expense_ratio: float
    fixed_expense_per_exposure: float | None
    profit_ratio: float
    permissible_loss_ratio: float


@dataclass(frozen=True)
class LossAdjustmentResult:
    """Result container for multiplicative loss adjustments."""

    input_total: float
    adjusted_total: float
    adjustment_factor: float
    components: dict[str, float]


@dataclass(frozen=True)
class RatedPolicyResult:
    """Result container for policy-level rating."""

    base_rate: float
    exposure: float
    relativity_product: float
    modifier_product: float
    variable_premium: float
    fixed_fee: float
    premium_before_min: float
    premium_after_min: float
    premium_final: float
