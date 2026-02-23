import pytest

from ratingengine.onlevel import average_rate_level, onlevel_factor, parallelogram_weights


def test_parallelogram_weights_and_arl_olf() -> None:
    changes = [
        {"effective_date": "2024-04-01", "change": 0.10},
        {"effective_date": "2024-10-01", "change": -0.05},
    ]
    weights = parallelogram_weights(changes, ("2024-01-01", "2025-01-01"))

    assert pytest.approx(weights["weight"].sum()) == 1.0

    arl = average_rate_level(weights["weight"], weights["rate_level"])
    expected_arl = (91.0 * 1.0 + 183.0 * 1.1 + 92.0 * 1.045) / 366.0
    assert arl == pytest.approx(expected_arl)

    olf = onlevel_factor(1.045, arl)
    assert olf == pytest.approx(1.045 / expected_arl)


def test_onlevel_factor_invalid() -> None:
    with pytest.raises(ValueError):
        onlevel_factor(0.0, 1.0)
    with pytest.raises(ValueError):
        onlevel_factor(1.0, 0.0)
