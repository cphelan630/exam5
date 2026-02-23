"""Pricing and ratemaking helper library for CAS Exam 5 Domain A."""

from .classratemaking import apply_rels, univariate_lr_rels, univariate_pp_rels
from .credibility import blend, limited_fluctuation_z
from .data_prep import add_time_columns, aggregate_experience, safe_divide
from .expenses import (
    expense_provisions_all_variable,
    expense_provisions_exposure_based,
    expense_provisions_premium_based,
    permissible_loss_ratio,
)
from .indication import explain_equivalence, loss_ratio_indication, pure_premium_indication
from .loss_adj import apply_loss_adjustment_factors, cap_losses, exclude_catastrophes
from .onlevel import average_rate_level, onlevel_factor, parallelogram_weights
from .overall_indication import indicated_premium, overall_indication
from .rating_manual import rate_book, rate_policy
from .trend import trend_to_average_date, two_step_trend
from .types import (
    ExpenseProvisionResult,
    LossAdjustmentResult,
    LossRatioIndicationResult,
    PurePremiumIndicationResult,
    RatedPolicyResult,
    TrendResult,
)

__all__ = [
    "ExpenseProvisionResult",
    "LossAdjustmentResult",
    "LossRatioIndicationResult",
    "PurePremiumIndicationResult",
    "RatedPolicyResult",
    "TrendResult",
    "add_time_columns",
    "aggregate_experience",
    "apply_loss_adjustment_factors",
    "apply_rels",
    "average_rate_level",
    "blend",
    "cap_losses",
    "exclude_catastrophes",
    "expense_provisions_all_variable",
    "expense_provisions_exposure_based",
    "expense_provisions_premium_based",
    "explain_equivalence",
    "indicated_premium",
    "limited_fluctuation_z",
    "loss_ratio_indication",
    "onlevel_factor",
    "overall_indication",
    "parallelogram_weights",
    "permissible_loss_ratio",
    "pure_premium_indication",
    "rate_book",
    "rate_policy",
    "safe_divide",
    "trend_to_average_date",
    "two_step_trend",
    "univariate_lr_rels",
    "univariate_pp_rels",
]
