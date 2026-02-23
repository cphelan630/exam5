import ratingengine


def test_public_api_exports() -> None:
    expected = [
        "parallelogram_weights",
        "average_rate_level",
        "onlevel_factor",
        "trend_to_average_date",
        "two_step_trend",
        "pure_premium_indication",
        "loss_ratio_indication",
        "explain_equivalence",
        "univariate_pp_rels",
        "univariate_lr_rels",
        "apply_rels",
        "limited_fluctuation_z",
        "blend",
        "permissible_loss_ratio",
        "rate_policy",
        "rate_book",
        "TrendResult",
        "PurePremiumIndicationResult",
    ]
    for name in expected:
        assert hasattr(ratingengine, name), name
