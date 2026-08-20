"""
Age-adjusts squad value: transfer value prices ability *and* resale potential,
and resale potential is heavily age-weighted (an 18- and a 29-year-old of
identical current ability get priced very differently), so raw squad_value is
a noisy proxy for current on-pitch quality -- part of its variance is just
squad age composition, not ability.

Fits log(value) ~ age + age^2 across every valued player-season in
transfermarkt_player_values (a clean inverted-U, peaking ~25-26, confirmed by
inspection before choosing this functional form). Each player's value is then
rebased to what they'd be worth at the curve's peak age, holding their
age-curve residual (how far above/below the typical value for their age --
the actual quality signal) fixed. Summing rebased values per team-season gives
an age-adjusted squad value that should be a cleaner predictor of on-pitch
performance than raw squad value, since it strips out the part of value driven
by "this squad happens to be young/old" rather than "this squad is good."

Values are CPI-adjusted to 2025-26 terms first (same deflator as
adjust_market_values_for_inflation.py) so the age curve is fit on a single
price level rather than mixing 2015 euros with 2025 euros.

Input:  transfermarkt_player_values table in infra/data/db/priors.db
Output: transfermarkt_squad_values_age_adjusted.csv, one row per team-season:
    team, season, n_players, avg_age,
    squad_value_2025_26_terms          (CPI-adjusted raw sum, for comparison)
    squad_value_age_adjusted_2025_26_terms
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from adjust_salaries_for_inflation import DB_PATH, TARGET_SEASON, season_avg_cpi

OUT_PATH = Path(__file__).parent / "transfermarkt_squad_values_age_adjusted.csv"


def fit_age_curve(df: pd.DataFrame) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, float]:
    """OLS log(value) ~ age + age^2 across all player-seasons. Returns the
    fitted model and the curve's peak age (where d/d(age) = 0)."""
    X = sm.add_constant(df[["age", "age_sq"]])
    model = sm.OLS(np.log(df["value_2025_26_terms"]), X).fit()
    b_age, b_age_sq = model.params["age"], model.params["age_sq"]
    peak_age = -b_age / (2 * b_age_sq)
    return model, peak_age


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transfermarkt_player_values", conn)
    conn.close()

    seasons = sorted(df["season"].unique())
    cpi_by_season = {s: season_avg_cpi(s) for s in seasons}
    target_cpi = cpi_by_season[TARGET_SEASON]
    df["deflator"] = df["season"].map(cpi_by_season).rdiv(target_cpi)
    df["value_2025_26_terms"] = df["value_eur"] * df["deflator"]
    df["age_sq"] = df["age"] ** 2

    model, peak_age = fit_age_curve(df)
    print(f"log(value) ~ age + age^2   n={len(df)}  R^2={model.rsquared:.3f}")
    for name, coef, se, p in zip(model.params.index, model.params, model.bse, model.pvalues):
        print(f"  {name:>8}: {coef: .4f}  (se={se:.4f}, p={p:.3f})")
    print(f"  peak age (curve maximum): {peak_age:.1f}")

    pred_log_value = model.predict(sm.add_constant(df[["age", "age_sq"]]))
    peak_row = sm.add_constant(pd.DataFrame({"age": [peak_age], "age_sq": [peak_age ** 2]}), has_constant="add")
    pred_log_peak = model.predict(peak_row)[0]
    # rebase to peak age: value * (curve at peak / curve at this age), i.e. keep
    # the age-curve residual (this player vs typical-for-age) fixed
    df["age_adjusted_value"] = df["value_2025_26_terms"] * np.exp(pred_log_peak - pred_log_value)

    squads = (
        df.groupby(["team", "season"])
        .agg(
            n_players=("player", "count"),
            avg_age=("age", "mean"),
            squad_value_2025_26_terms=("value_2025_26_terms", "sum"),
            squad_value_age_adjusted_2025_26_terms=("age_adjusted_value", "sum"),
        )
        .reset_index()
    )
    squads["avg_age"] = squads["avg_age"].round(1)
    squads[["squad_value_2025_26_terms", "squad_value_age_adjusted_2025_26_terms"]] = (
        squads[["squad_value_2025_26_terms", "squad_value_age_adjusted_2025_26_terms"]].round().astype(int)
    )

    squads.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(squads)} team-seasons -> {OUT_PATH}")
