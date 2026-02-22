"""Basic overall indication calculations."""


# def indicated_premium(losses: float, expenses: float, profit: float = 0.0) -> float:
#     """Premium = Losses + Expenses + Profit."""
#     return losses + expenses + profit


# def overall_indication(
#     current_premium: float, losses: float, expenses: float, profit: float = 0.0
# ) -> float:
#     """Return indicated overall rate change as a decimal."""
#     if current_premium <= 0:
#         raise ValueError("current_premium must be greater than 0")

#     indicated = indicated_premium(losses=losses, expenses=expenses, profit=profit)
#     return (indicated / current_premium) - 1.0