"""Overall indication calculations (pure premium and loss ratio methods)."""

from __future__ import annotations

import math

from .types import LossRatioIndicationResult, PurePremiumIndicationResult


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


def pure_premium_indication(
    losses: float,
    lae: float,
    exposures: float,
    fixed_exp_per_exp: float,
    var_exp_ratio: float,
    profit_ratio: float,
    *,
    current_premium_per_exposure: float | None = None,
) -> PurePremiumIndicationResult:
    """Compute a pure premium indication."""

    losses_val = float(losses)
    lae_val = float(lae)
    exposures_val = _validate_positive(exposures, name="exposures")
    fixed_exp_val = float(fixed_exp_per_exp)
    var_exp_val = _validate_ratio(var_exp_ratio, name="var_exp_ratio")
    profit_val = _validate_ratio(profit_ratio, name="profit_ratio")

    denominator = 1.0 - var_exp_val - profit_val
    if denominator <= 1e-12:
        raise ValueError("var_exp_ratio + profit_ratio must be less than 1.")

    loss_plus_lae = losses_val + lae_val
    loss_plus_lae_per_exposure = loss_plus_lae / exposures_val
    indicated_pp = (loss_plus_lae_per_exposure + fixed_exp_val) / denominator

    current_pp: float | None = None
    indicated_change: float | None = None
    if current_premium_per_exposure is not None:
        current_pp = _validate_positive(
            current_premium_per_exposure,
            name="current_premium_per_exposure",
        )
        indicated_change = indicated_pp / current_pp - 1.0

    return PurePremiumIndicationResult(
        losses=losses_val,
        lae=lae_val,
        exposures=exposures_val,
        loss_plus_lae=loss_plus_lae,
        loss_plus_lae_per_exposure=loss_plus_lae_per_exposure,
        fixed_exp_per_exp=fixed_exp_val,
        var_exp_ratio=var_exp_val,
        profit_ratio=profit_val,
        denominator=denominator,
        indicated_premium_per_exposure=indicated_pp,
        current_premium_per_exposure=current_pp,
        indicated_change=indicated_change,
    )


def loss_ratio_indication(
    ultimate_losses: float,
    lae: float,
    earned_prem_onlevel: float,
    fixed_exp_ratio: float,
    var_exp_ratio: float,
    profit_ratio: float,
) -> LossRatioIndicationResult:
    """Compute a loss ratio indication."""

    losses_val = float(ultimate_losses)
    lae_val = float(lae)
    premium_val = _validate_positive(earned_prem_onlevel, name="earned_prem_onlevel")
    fixed_exp_val = _validate_ratio(fixed_exp_ratio, name="fixed_exp_ratio")
    var_exp_val = _validate_ratio(var_exp_ratio, name="var_exp_ratio")
    profit_val = _validate_ratio(profit_ratio, name="profit_ratio")

    plr = 1.0 - fixed_exp_val - var_exp_val - profit_val
    if plr <= 1e-12:
        raise ValueError(
            "fixed_exp_ratio + var_exp_ratio + profit_ratio must be less than 1."
        )

    actual_lr = (losses_val + lae_val) / premium_val
    indicated_change = actual_lr / plr - 1.0

    return LossRatioIndicationResult(
        ultimate_losses=losses_val,
        lae=lae_val,
        earned_premium_onlevel=premium_val,
        actual_loss_ratio=actual_lr,
        fixed_exp_ratio=fixed_exp_val,
        var_exp_ratio=var_exp_val,
        profit_ratio=profit_val,
        permissible_loss_ratio=plr,
        indicated_change=indicated_change,
    )


def explain_equivalence() -> str:
    """Return a markdown explanation of PP and LR equivalence."""

    return (
        "## Pure Premium vs Loss Ratio Equivalence\n\n"
        "Let:\n"
        "- `L` = projected losses + LAE\n"
        "- `E` = exposures\n"
        "- `P` = premium at current rate level\n"
        "- `v` = variable expense ratio\n"
        "- `f` = fixed expense ratio\n"
        "- `F` = fixed expense per exposure\n"
        "- `q` = profit ratio\n\n"
        "Pure premium form:\n"
        "`PP_ind = (L / E + F) / (1 - v - q)`\n\n"
        "Loss ratio form:\n"
        "`Indicated Change = (L / P) / (1 - f - v - q) - 1`\n\n"
        "If fixed expenses are represented consistently (`F * E = f * P`), then both "
        "methods produce the same indicated premium level and therefore the same indicated "
        "rate change.\n"
    )
