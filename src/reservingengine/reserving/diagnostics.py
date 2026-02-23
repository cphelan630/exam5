"""Diagnostics for Triangle quality and development assumptions.

Covers the standard exhibits from Friedland Chapter 8:
  - Full age-to-age factor exhibit (link_ratio_table)
  - Heatmap of link ratios (plot_link_ratio_heatmap)
  - Calendar year / diagonal effects test
  - Paid vs incurred development comparison
  - Mack-style correlation tests
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .triangle_io import _import_chainladder, triangle_to_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triangle_to_wide(triangle: Any) -> pd.DataFrame:
    """Convert a single-column Triangle to a wide DataFrame (origins × dev ages)."""
    frame = triangle_to_frame(triangle, origin_as_datetime=False)
    reset = frame.reset_index()
    value_cols = [c for c in reset.columns if c not in {"origin", "development", "valuation"}
                  and not c.startswith("level_")]
    value_col = value_cols[-1] if value_cols else reset.columns[-1]
    wide = (
        reset.pivot(index="origin", columns="development", values=value_col)
        .sort_index()
        .sort_index(axis=1)
        .astype(float)
    )
    wide.index = wide.index.map(str)
    wide.columns = wide.columns.map(str)
    return wide


def _medial_average(values: pd.Series) -> float:
    """Volume-weighted medial: drop the single highest and lowest, then simple-average the rest."""
    valid = values.dropna()
    if len(valid) <= 2:
        return float(valid.mean())
    trimmed = valid.sort_values().iloc[1:-1]
    return float(trimmed.mean())


def _origin_year_numeric(origin_str: str) -> int | None:
    """Parse an origin label to a calendar year integer, or None if unparseable."""
    s = str(origin_str).strip()
    try:
        return int(s[:4])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Basic sanity check
# ---------------------------------------------------------------------------

def triangle_sanity_checks(triangle: Any) -> pd.DataFrame:
    """Return basic triangle diagnostics for quick quality checks."""

    frame = triangle_to_frame(triangle, origin_as_datetime=False)
    value_cols = [c for c in frame.columns if c not in {"origin", "development", "valuation"}]
    value_col = "values" if "values" in value_cols else value_cols[-1]
    values = pd.to_numeric(frame[value_col], errors="coerce")
    checks = {
        "rows": int(len(frame)),
        "non_null": int(values.notna().sum()),
        "null": int(values.isna().sum()),
        "min": float(np.nanmin(values.to_numpy(dtype=float))),
        "max": float(np.nanmax(values.to_numpy(dtype=float))),
        "total": float(np.nansum(values.to_numpy(dtype=float))),
    }
    return pd.DataFrame([checks])


# ---------------------------------------------------------------------------
# Link-ratio / age-to-age factor exhibit (Friedland Chapter 8 Exhibit 1)
# ---------------------------------------------------------------------------

def link_ratio_table(triangle: Any, *, n_periods: int = -1) -> pd.DataFrame:
    """Return the Friedland-style age-to-age factor exhibit as a wide DataFrame.

    Rows: individual origin years, then summary rows
    Columns: development age transitions ("12-24", "24-36", …)

    Summary rows appended:
      - "Vol Wtd Avg"  : volume-weighted average (the standard chainladder LDF)
      - "Simple Avg"   : arithmetic mean of all available factors
      - "Medial Avg"   : trimmed mean (drop highest and lowest, average the rest)
      - "Selected LDF" : result of cl.Development(n_periods=n_periods) fit

    Parameters
    ----------
    triangle:
        A single-column cumulative chainladder Triangle.
    n_periods:
        Passed to cl.Development for the selected-LDF row (-1 = all years).
    """
    cl = _import_chainladder()

    wide = _triangle_to_wide(triangle)
    dev_ages = list(wide.columns)
    transitions = [f"{dev_ages[i]}-{dev_ages[i+1]}" for i in range(len(dev_ages) - 1)]

    # Compute individual link ratios
    factor_rows: dict[str, dict[str, float | None]] = {}
    for origin in wide.index:
        row: dict[str, float | None] = {}
        for i, t in enumerate(transitions):
            num = wide.iloc[wide.index.get_loc(origin), i + 1]
            den = wide.iloc[wide.index.get_loc(origin), i]
            if pd.notna(num) and pd.notna(den) and den != 0:
                row[t] = float(num / den)
            else:
                row[t] = None
        factor_rows[origin] = row

    factor_df = pd.DataFrame.from_dict(factor_rows, orient="index")[transitions]
    factor_df.index.name = "origin"

    # Volume-weighted averages
    vol_wtd: dict[str, float | None] = {}
    for i, t in enumerate(transitions):
        num_col = wide.iloc[:, i + 1]
        den_col = wide.iloc[:, i]
        valid = num_col.notna() & den_col.notna() & (den_col != 0)
        if valid.any():
            vol_wtd[t] = float(num_col[valid].sum() / den_col[valid].sum())
        else:
            vol_wtd[t] = None

    # Simple averages (per transition column)
    simple_avg = {t: float(factor_df[t].mean()) if factor_df[t].notna().any() else None
                  for t in transitions}

    # Medial averages
    medial_avg = {t: _medial_average(factor_df[t]) if factor_df[t].notna().sum() >= 3 else simple_avg[t]
                  for t in transitions}

    # Selected LDFs from chainladder Development
    dev = cl.Development(n_periods=n_periods).fit(triangle)
    ldf_frame = dev.ldf_.to_frame(origin_as_datetime=False, keepdims=True).reset_index()
    value_col = [c for c in ldf_frame.columns if c not in {"origin", "development", "valuation"}][-1]
    selected_map = dict(zip(ldf_frame["development"].map(str), ldf_frame[value_col].astype(float)))
    selected_ldf = {t: selected_map.get(dev_ages[i + 1]) for i, t in enumerate(transitions)}

    summary = pd.DataFrame(
        [vol_wtd, simple_avg, medial_avg, selected_ldf],
        index=["Vol Wtd Avg", "Simple Avg", "Medial Avg", "Selected LDF"],
        columns=transitions,
    )
    summary.index.name = "origin"

    return pd.concat([factor_df, summary])


def plot_link_ratio_heatmap(
    triangle: Any,
    *,
    n_periods: int = -1,
    figsize: tuple[int, int] = (12, 7),
    cmap: str = "RdYlGn_r",
    fmt: str = ".3f",
    title: str = "Age-to-Age Development Factor Heatmap",
    ax: Any = None,
) -> Any:
    """Plot a heatmap of all age-to-age link ratios (origins × transitions).

    Cells are colored relative to the column median: green below median (fast
    development), red above (slow development).  Summary rows are excluded from
    the color scale.

    Returns the matplotlib Axes object.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib and seaborn are required for plot_link_ratio_heatmap."
        ) from exc

    table = link_ratio_table(triangle, n_periods=n_periods)

    summary_labels = {"Vol Wtd Avg", "Simple Avg", "Medial Avg", "Selected LDF"}
    factor_only = table.loc[~table.index.isin(summary_labels)].astype(float)

    # Center colormap at column median so deviations are visually symmetric
    center = float(factor_only.median().mean())

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        factor_only,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "Link Ratio"},
    )
    ax.set_title(title)
    ax.set_xlabel("Development Age Transition")
    ax.set_ylabel("Origin Year")
    plt.tight_layout()
    return ax


def selected_ldf_table(triangle: Any, **development_kwargs: Any) -> pd.DataFrame:
    """Fit Development and return selected LDFs as a tidy table."""

    cl = _import_chainladder()
    dev = cl.Development(**development_kwargs).fit(triangle)
    ldf = dev.ldf_.to_frame(origin_as_datetime=False, keepdims=True).reset_index()
    if "values" in ldf.columns:
        ldf = ldf.rename(columns={"values": "selected_ldf"})
    return ldf


# ---------------------------------------------------------------------------
# Calendar year (diagonal) effects diagnostic  (Friedland Chapter 8)
# ---------------------------------------------------------------------------

def calendar_year_diagnostic(triangle: Any) -> dict[str, pd.DataFrame]:
    """Test for calendar year effects in age-to-age development factors.

    Implements the L/S (Large/Small) test described by Friedland:
      - For each development period, label each year's factor "L" (above
        column median) or "S" (at or below column median).
      - Map each factor to its calendar year (origin + later development age).
      - Summarise by calendar year: count of L's and S's across all
        development periods active in that year.

    Returns a dict with two DataFrames:
      ``"ls_matrix"``
          Wide matrix of "L" / "S" / NaN labels (origins × transitions).
      ``"calendar_year_summary"``
          Summary with columns: CalendarYear, n_L, n_S, n_total, pct_L,
          interpretation.  A year with very high pct_L suggests above-average
          development (possible superimposed inflation); very low pct_L
          suggests below-average development.
    """
    wide = _triangle_to_wide(triangle)
    dev_ages_raw = list(wide.columns)
    dev_ages_num: list[float] = []
    for a in dev_ages_raw:
        try:
            dev_ages_num.append(float(a))
        except ValueError:
            dev_ages_num.append(float("nan"))

    transitions = [f"{dev_ages_raw[i]}-{dev_ages_raw[i+1]}" for i in range(len(dev_ages_raw) - 1)]

    # Build factor matrix
    factor_data: dict[str, dict[str, float | None]] = {}
    for origin in wide.index:
        row: dict[str, float | None] = {}
        for i, t in enumerate(transitions):
            num = wide.iloc[wide.index.get_loc(origin), i + 1]
            den = wide.iloc[wide.index.get_loc(origin), i]
            if pd.notna(num) and pd.notna(den) and den != 0:
                row[t] = float(num / den)
            else:
                row[t] = None
        factor_data[origin] = row

    factor_df = pd.DataFrame.from_dict(factor_data, orient="index")[transitions]
    factor_df.index.name = "origin"

    # L/S labels (above vs. at-or-below column median)
    ls_rows: dict[str, dict[str, str | None]] = {}
    for origin in factor_df.index:
        ls_row: dict[str, str | None] = {}
        for t in transitions:
            val = factor_df.at[origin, t]
            if pd.isna(val):
                ls_row[t] = None
                continue
            col_median = float(factor_df[t].median())
            ls_row[t] = "L" if val > col_median else "S"
        ls_rows[origin] = ls_row

    ls_matrix = pd.DataFrame.from_dict(ls_rows, orient="index")[transitions]
    ls_matrix.index.name = "origin"

    # Map each cell to its calendar year
    #   calendar_year ≈ origin_year + later_dev_age (in same units)
    cy_records: list[dict[str, Any]] = []
    for origin in ls_matrix.index:
        origin_yr = _origin_year_numeric(origin)
        if origin_yr is None:
            continue
        for i, t in enumerate(transitions):
            label = ls_matrix.at[origin, t]
            if label is None:
                continue
            later_age = dev_ages_num[i + 1]
            if not np.isnan(later_age):
                # If ages are in months, convert to years; if already years, use directly
                age_years = later_age / 12.0 if later_age > 24 else later_age
                cy = int(origin_yr + age_years - 1)
            else:
                cy = origin_yr
            cy_records.append({"calendar_year": cy, "label": label})

    cy_df = pd.DataFrame(cy_records)
    if cy_df.empty:
        summary = pd.DataFrame(
            columns=["CalendarYear", "n_L", "n_S", "n_total", "pct_L", "interpretation"]
        )
    else:
        grouped = cy_df.groupby("calendar_year")["label"]
        n_L = grouped.apply(lambda s: (s == "L").sum()).rename("n_L")
        n_S = grouped.apply(lambda s: (s == "S").sum()).rename("n_S")
        summary = pd.concat([n_L, n_S], axis=1).reset_index()
        summary.columns = pd.Index(["CalendarYear", "n_L", "n_S"])
        summary["n_total"] = summary["n_L"] + summary["n_S"]
        summary["pct_L"] = np.where(
            summary["n_total"] > 0,
            summary["n_L"] / summary["n_total"],
            np.nan,
        )

        def _interpret(row: pd.Series) -> str:
            if pd.isna(row["pct_L"]) or row["n_total"] < 2:
                return "insufficient data"
            p = row["pct_L"]
            if p >= 0.75:
                return "above-average development (possible superimposed inflation)"
            if p <= 0.25:
                return "below-average development"
            return "no strong calendar year signal"

        summary["interpretation"] = summary.apply(_interpret, axis=1)

    return {
        "ls_matrix": ls_matrix,
        "calendar_year_summary": summary.set_index("CalendarYear")
        if not summary.empty
        else summary,
    }


# ---------------------------------------------------------------------------
# Paid vs incurred development comparison  (Friedland Chapter 8)
# ---------------------------------------------------------------------------

def paid_vs_incurred_comparison(
    paid_triangle: Any,
    incurred_triangle: Any,
    *,
    n_periods: int = -1,
) -> dict[str, pd.DataFrame]:
    """Compare development patterns between paid and incurred (reported) triangles.

    Returns a dict with:
      ``"ldf_comparison"``
          Side-by-side selected LDFs: columns ``paid_ldf``, ``incurred_ldf``,
          ``ldf_ratio`` (paid / incurred; values < 1 mean paid develops faster
          than incurred at that age, which is expected and normal).
      ``"cdf_comparison"``
          Same structure for cumulative development factors (CDFs to ultimate).
      ``"paid_to_incurred_latest"``
          Latest-diagonal paid-to-incurred ratio by origin year.  Should
          approach 1.0 as each year matures.  Persistent divergence may signal
          case reserve adequacy changes.
    """
    cl = _import_chainladder()

    dev_paid = cl.Development(n_periods=n_periods).fit(paid_triangle)
    dev_incurred = cl.Development(n_periods=n_periods).fit(incurred_triangle)

    def _pattern_series(dev_model: Any, attr: str) -> pd.Series:
        tri = getattr(dev_model, attr)
        frame = tri.to_frame(origin_as_datetime=False, keepdims=True).reset_index()
        value_col = [c for c in frame.columns
                     if c not in {"origin", "development", "valuation"}][-1]
        return pd.Series(
            frame[value_col].astype(float).values,
            index=frame["development"].map(str),
            dtype=float,
        )

    paid_ldf = _pattern_series(dev_paid, "ldf_")
    incurred_ldf = _pattern_series(dev_incurred, "ldf_")
    paid_cdf = _pattern_series(dev_paid, "cdf_")
    incurred_cdf = _pattern_series(dev_incurred, "cdf_")

    shared_ldf_idx = paid_ldf.index.union(incurred_ldf.index)
    ldf_comp = pd.DataFrame(
        {
            "paid_ldf": paid_ldf.reindex(shared_ldf_idx),
            "incurred_ldf": incurred_ldf.reindex(shared_ldf_idx),
        }
    )
    ldf_comp["ldf_ratio"] = ldf_comp["paid_ldf"] / ldf_comp["incurred_ldf"]
    ldf_comp.index.name = "development_age"

    shared_cdf_idx = paid_cdf.index.union(incurred_cdf.index)
    cdf_comp = pd.DataFrame(
        {
            "paid_cdf": paid_cdf.reindex(shared_cdf_idx),
            "incurred_cdf": incurred_cdf.reindex(shared_cdf_idx),
        }
    )
    cdf_comp["cdf_ratio"] = cdf_comp["paid_cdf"] / cdf_comp["incurred_cdf"]
    cdf_comp.index.name = "development_age"

    # Latest-diagonal paid / incurred by origin
    paid_wide = _triangle_to_wide(paid_triangle)
    incurred_wide = _triangle_to_wide(incurred_triangle)
    shared_origins = paid_wide.index.intersection(incurred_wide.index)

    latest_paid = paid_wide.loc[shared_origins].apply(
        lambda row: row.dropna().iloc[-1] if row.notna().any() else np.nan, axis=1
    )
    latest_incurred = incurred_wide.loc[shared_origins].apply(
        lambda row: row.dropna().iloc[-1] if row.notna().any() else np.nan, axis=1
    )
    ratio_series = (latest_paid / latest_incurred).rename("paid_to_incurred")
    pi_df = pd.DataFrame(
        {
            "latest_paid": latest_paid,
            "latest_incurred": latest_incurred,
            "paid_to_incurred": ratio_series,
        }
    )
    pi_df.index.name = "origin"

    return {
        "ldf_comparison": ldf_comp,
        "cdf_comparison": cdf_comp,
        "paid_to_incurred_latest": pi_df,
    }


# ---------------------------------------------------------------------------
# Additional diagnostics for paid/reported, trends, counts, case adequacy
# ---------------------------------------------------------------------------

def percent_paid_reported(triangle: Any) -> pd.DataFrame:
    """Return latest & ultimate values and percent-paid for each origin.

    The calculation uses a Chainladder fit on ``triangle`` to obtain
    ultimate estimates.  Returned DataFrame has columns ``latest``,
    ``ultimate``, ``ibnr`` (ultimate minus latest), ``percent_paid``
    and ``percent_unreported``.  Origin labels are preserved as the index.
    """

    cl = _import_chainladder()
    # fit chainladder to get ultimates/ibnr
    model = cl.Chainladder().fit(triangle)

    latest = triangle.latest_diagonal.to_frame(origin_as_datetime=False, keepdims=True)
    latest = latest.rename(columns={latest.columns[-1]: "latest"})

    ultimate = model.ultimate_.to_frame(origin_as_datetime=False, keepdims=True)
    ultimate = ultimate.rename(columns={ultimate.columns[-1]: "ultimate"})

    ibnr = model.ibnr_.to_frame(origin_as_datetime=False, keepdims=True)
    ibnr = ibnr.rename(columns={ibnr.columns[-1]: "ibnr"})

    df = latest.join(ultimate).join(ibnr)
    df.index.name = "origin"
    df["percent_paid"] = df["latest"] / df["ultimate"]
    df["percent_unreported"] = 1.0 - df["percent_paid"]
    return df


def projection_comparison(
    paid_triangle: Any,
    reported_triangle: Any,
    **chainladder_kwargs: Any,
) -> pd.DataFrame:
    """Compare chain‐ladder projections side‑by‑side for paid vs reported.

    Fits ``Chainladder(**chainladder_kwargs)`` to each input triangle and
    returns a single wide DataFrame indexed by origin with the following
    columns (multi‑level):

    ``('paid',  'latest')``  latest diagonal paid value
    ``('paid',  'ultimate')`` ultimate from model
    ``('paid',  'ibnr')``    ibnr from model
    ``('reported','latest')``  same for reported triangle
    ``('reported','ultimate')``
    ``('reported','ibnr')``
    ``paid_to_reported_ultimate``   ratio of ultimate amounts (scalar for each origin)
    """
    cl = _import_chainladder()

    def _extract(tri: Any, model: Any) -> pd.DataFrame:
        latest = tri.latest_diagonal.to_frame(origin_as_datetime=False, keepdims=True)
        latest = latest.rename(columns={latest.columns[-1]: "latest"})
        ultimate = model.ultimate_.to_frame(origin_as_datetime=False, keepdims=True)
        ultimate = ultimate.rename(columns={ultimate.columns[-1]: "ultimate"})
        ibnr = model.ibnr_.to_frame(origin_as_datetime=False, keepdims=True)
        ibnr = ibnr.rename(columns={ibnr.columns[-1]: "ibnr"})
        df = latest.join(ultimate).join(ibnr)
        return df

    paid_model = cl.Chainladder(**chainladder_kwargs).fit(paid_triangle)
    rep_model = cl.Chainladder(**chainladder_kwargs).fit(reported_triangle)

    paid_df = _extract(paid_triangle, paid_model)
    rep_df = _extract(reported_triangle, rep_model)

    comp = pd.concat({"paid": paid_df, "reported": rep_df}, axis=1)
    comp.index.name = "origin"
    # compute ratio column at top level
    comp["paid_to_reported_ultimate"] = (
        comp[("paid", "ultimate")] / comp[("reported", "ultimate")]
    )
    return comp


def trend_summary(triangle: Any) -> dict[str, pd.DataFrame]:
    """Simple slope‑based trend diagnostics for a single triangle.

    Returns a dict containing three DataFrames/Series:
      * ``age_trends`` – slope of link ratios across development ages for each origin
      * ``origin_trends`` – slope of link ratios across origin years for each age
      * ``diagonal_trend`` – a one‑row DataFrame with the slope of average
        link ratio by calendar year (diagonal).
    """
    wide = _triangle_to_wide(triangle)
    # numeric versions of origins and ages for regression
    origins_num = pd.Series(
        [(_origin_year_numeric(o) or np.nan) for o in wide.index],
        index=wide.index,
        dtype=float,
    )
    ages_num = pd.Series(
        [float(str(a)) if _is_number(str(a)) else np.nan for a in wide.columns],
        index=wide.columns,
        dtype=float,
    )

    # build link ratios matrix (same shape as wide, NaN first column)
    ratios = wide.copy()
    for i in range(1, len(wide.columns)):
        ratios.iloc[:, i] = wide.iloc[:, i] / wide.iloc[:, i - 1]
    ratios = ratios.iloc[:, 1:]

    # helper for slope
    def _slope(x: pd.Series, y: pd.Series) -> float:
        x = x.dropna()
        y = y.loc[x.index].dropna()
        if len(x) >= 2 and len(y) >= 2:
            return float(np.polyfit(x.values.astype(float), y.values.astype(float), 1)[0])
        return float(np.nan)

    age_trends = {}
    for origin in ratios.index:
        row = ratios.loc[origin].dropna()
        age_trends[origin] = _slope(ages_num.reindex(row.index), row)

    origin_trends = {}
    for age in ratios.columns:
        col = ratios[age].dropna()
        origin_trends[age] = _slope(origins_num.reindex(col.index), col)

    # diagonal trend – average link ratio by calendar year
    cy_records: list[dict[str, float]] = []
    dev_ages = list(wide.columns)
    dev_ages_num = [float(str(a)) if _is_number(str(a)) else np.nan for a in dev_ages]
    for origin in ratios.index:
        orig_yr = _origin_year_numeric(origin)
        if orig_yr is None:
            continue
        for idx, age in enumerate(ratios.columns):
            val = ratios.at[origin, age]
            if pd.isna(val):
                continue
            later_age = dev_ages_num[idx + 1] if idx + 1 < len(dev_ages_num) else np.nan
            if not np.isnan(later_age):
                age_years = later_age / 12.0 if later_age > 24 else later_age
                cy = int(orig_yr + age_years - 1)
            else:
                cy = orig_yr
            cy_records.append({"calendar_year": cy, "link_ratio": val})
    cy_df = pd.DataFrame(cy_records)
    if not cy_df.empty:
        mean_by_year = cy_df.groupby("calendar_year")["link_ratio"].mean()
        diag_slope = _slope(mean_by_year.reset_index()["calendar_year"], mean_by_year)
    else:
        diag_slope = float(np.nan)
    diagonal_trend = pd.DataFrame({"slope": [diag_slope]}, index=["calendar_year"])

    return {
        "age_trends": pd.Series(age_trends, name="slope"),
        "origin_trends": pd.Series(origin_trends, name="slope"),
        "diagonal_trend": diagonal_trend,
    }


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def count_triangle_diagnostics(triangle: Any) -> dict[str, Any]:
    """Diagnostics tailored to claim‑count triangles.

    Returns a dict with:
      ``"link_ratio_table"`` – same output as :func:`link_ratio_table`.
      ``"percent_closed_by_age"`` – Series giving 1 − 1/(vol‑wtd) factor.
    """
    lr = link_ratio_table(triangle)
    if "Vol Wtd Avg" in lr.index:
        vol = lr.loc["Vol Wtd Avg"].astype(float)
        pct_closed = 1.0 - 1.0 / vol
    else:
        pct_closed = pd.Series(dtype=float)
    return {"link_ratio_table": lr, "percent_closed_by_age": pct_closed}


def case_adequacy_indicator(
    paid_triangle: Any, incurred_triangle: Any
) -> pd.DataFrame:
    """Quick indicator of case reserve adequacy trends.

    Uses :func:`paid_vs_incurred_comparison` to obtain the latest paid‑to‑
    incurred ratios and computes the slope of that series over origin years.
    """
    comp = paid_vs_incurred_comparison(paid_triangle, incurred_triangle)
    ratio = comp["paid_to_incurred_latest"]["paid_to_incurred"]
    orig_nums = pd.Series([_origin_year_numeric(o) for o in ratio.index], index=ratio.index)
    slope = float(
        np.polyfit(orig_nums.dropna().values.astype(float), ratio.dropna().values.astype(float), 1)[0]
    ) if ratio.dropna().shape[0] >= 2 else float(np.nan)
    df = pd.DataFrame({"paid_to_incurred": ratio})
    df["trend_slope"] = slope
    return df


# ---------------------------------------------------------------------------
# Correlation tests  (Mack-style, already present)
# ---------------------------------------------------------------------------

def run_correlation_tests(
    triangle: Any,
    *,
    p_critical_development: float = 0.5,
    p_critical_valuation: float = 0.1,
    valuation_total: bool = True,
) -> dict[str, Any]:
    """Run Mack-style development and valuation correlation diagnostics."""

    cl = _import_chainladder()

    dev_corr = cl.DevelopmentCorrelation(triangle, p_critical=p_critical_development)
    val_corr = cl.ValuationCorrelation(
        triangle,
        p_critical=p_critical_valuation,
        total=valuation_total,
    )
    return {
        "development_correlation": dev_corr,
        "valuation_correlation": val_corr,
    }
