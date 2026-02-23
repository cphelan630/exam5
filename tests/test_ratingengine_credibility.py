from statistics import NormalDist

import pytest

from ratingengine.credibility import blend, limited_fluctuation_z


def test_limited_fluctuation_with_n_full_and_clipping() -> None:
    assert limited_fluctuation_z(25.0, 100.0) == pytest.approx(0.5)
    assert limited_fluctuation_z(400.0, 100.0) == pytest.approx(1.0)


def test_limited_fluctuation_when_n_full_is_none() -> None:
    z_crit = NormalDist().inv_cdf((1.0 + 0.95) / 2.0)
    n_full_calc = (z_crit / 0.05) ** 2
    expected = min(1.0, (64.0 / n_full_calc) ** 0.5)
    assert limited_fluctuation_z(64.0, None, prob=0.95, error=0.05) == pytest.approx(expected)


def test_blend_and_validation() -> None:
    assert blend(0.4, 0.10, 0.02) == pytest.approx(0.052)
    with pytest.raises(ValueError):
        blend(1.2, 0.1, 0.0)
