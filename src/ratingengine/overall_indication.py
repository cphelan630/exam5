"""Compatibility helpers for basic overall indication calculations.

For richer, exam-style workflows, use :mod:`ratingengine.indication`.
"""

from __future__ import annotations


def indicated_premium(losses: float, expenses: float, profit: float = 0.0) -> float:
    """Compute indicated premium as losses + expenses + profit."""

    return float(losses) + float(expenses) + float(profit)


def overall_indication(
    current_premium: float,
    losses: float,
    expenses: float,
    profit: float = 0.0,
) -> float:
    """Return indicated overall rate change as a decimal."""

    current = float(current_premium)
    if current <= 0.0:
        raise ValueError("current_premium must be greater than 0.")

    indicated = indicated_premium(losses=losses, expenses=expenses, profit=profit)
    return indicated / current - 1.0
