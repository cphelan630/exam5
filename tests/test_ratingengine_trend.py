import pandas as pd
import pytest

from ratingengine.trend import trend_to_average_date, two_step_trend


def test_trend_to_average_date_one_year_like_span() -> None:
    result = trend_to_average_date(
        value=100.0,
        from_date="2024-01-01",
        to_date="2025-01-01",
        annual_rate=0.10,
    )
    years = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days / 365.25
    expected_factor = (1.10) ** years
    assert result.total_factor == pytest.approx(expected_factor)
    assert result.trended_level == pytest.approx(100.0 * expected_factor)


def test_two_step_trend_with_explicit_as_of_date() -> None:
    result = two_step_trend(
        base_level=200.0,
        mid_period_date="2023-07-01",
        target_date="2025-07-01",
        retro_trend=0.06,
        pros_trend=0.04,
        as_of_date="2024-07-01",
    )
    years_retro = (pd.Timestamp("2024-07-01") - pd.Timestamp("2023-07-01")).days / 365.25
    years_pros = (pd.Timestamp("2025-07-01") - pd.Timestamp("2024-07-01")).days / 365.25
    expected_factor = (1.06**years_retro) * (1.04**years_pros)
    assert result.years_retro == pytest.approx(years_retro)
    assert result.years_pros == pytest.approx(years_pros)
    assert result.total_factor == pytest.approx(expected_factor)
    assert result.trended_level == pytest.approx(200.0 * expected_factor)


def test_two_step_trend_date_parsing_and_validation() -> None:
    result = two_step_trend(
        base_level=100.0,
        mid_period_date=pd.Timestamp("2024-01-01"),
        target_date=pd.Timestamp("2025-01-01"),
        retro_trend=0.0,
        pros_trend=0.0,
        as_of_date="2024-07-01",
    )
    assert result.total_factor == pytest.approx(1.0)

    with pytest.raises(ValueError):
        two_step_trend(
            base_level=100.0,
            mid_period_date="2025-01-01",
            target_date="2024-01-01",
            retro_trend=0.0,
            pros_trend=0.0,
            as_of_date="2024-06-01",
        )
