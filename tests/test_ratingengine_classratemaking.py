import pandas as pd
import pytest

from ratingengine.classratemaking import apply_rels, univariate_lr_rels, univariate_pp_rels


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": ["A", "B"],
            "loss": [90.0, 30.0],
            "exposure": [30.0, 20.0],
            "premium": [60.0, 40.0],
        }
    )


def test_univariate_pp_and_lr_relativities() -> None:
    df = _toy_df()

    pp = univariate_pp_rels(df, "class", "loss", "exposure").set_index("class_value")
    assert pp.loc["A", "metric"] == pytest.approx(3.0)
    assert pp.loc["B", "metric"] == pytest.approx(1.5)
    assert pp.loc["A", "relativity"] == pytest.approx(1.25)
    assert pp.loc["B", "relativity"] == pytest.approx(0.625)

    lr = univariate_lr_rels(df, "class", "loss", "premium").set_index("class_value")
    assert lr.loc["A", "metric"] == pytest.approx(1.5)
    assert lr.loc["B", "metric"] == pytest.approx(0.75)
    assert lr.loc["A", "relativity"] == pytest.approx(1.25)
    assert lr.loc["B", "relativity"] == pytest.approx(0.625)


def test_apply_rels_with_dict_series_and_dataframe() -> None:
    rel_dict = {"A": 1.25, "B": 0.625}
    out_dict = apply_rels(100.0, rel_dict, fixed_fee=10.0)
    assert out_dict.loc["A"] == pytest.approx(135.0)
    assert out_dict.loc["B"] == pytest.approx(72.5)

    rel_series = pd.Series(rel_dict)
    out_series = apply_rels(100.0, rel_series, fixed_fee=10.0)
    assert out_series.loc["A"] == pytest.approx(135.0)

    rel_df = pd.DataFrame({"rel": rel_dict})
    out_df = apply_rels(100.0, rel_df, fixed_fee=10.0)
    assert out_df.loc["B"] == pytest.approx(72.5)
