"""
Two simple OLS regressions, same-season PL scoring across all PL team-seasons:
     pl_gf_rate ~ log(gross_2025_26_terms)      (ATTACK, wage bill)
     pl_ga_rate ~ log(gross_2025_26_terms)      (DEFENSE, wage bill)
     pl_gf_rate ~ log(squad_value_2025_26_terms)  (ATTACK, squad value)
     pl_ga_rate ~ log(squad_value_2025_26_terms)  (DEFENSE, squad value)
Uses prem_salaries_adjusted.csv / transfermarkt_squad_values_adjusted.csv
(CPI-adjusted, see adjust_salaries_for_inflation.py /
adjust_market_values_for_inflation.py) joined to each team's own-season PL
goals rate from the raw_results table. Both predictors are log-transformed
since spending spans two orders of magnitude and marginal returns taper off.

This module originally also had a Championship-form -> promoted-team-scoring
regression here (against raw_results goals, which include penalties -- a
mismatch with the non_penalty_bayes model). That's been superseded by
promotion/champ_analysis.ipynb's non-penalty version sourced directly from
fotmob.db, and removed from here along with its now-deleted input file
(champ_performance.csv).

Team names differ between prem_salaries_adjusted.csv and raw_results (the
latter uses football-data.co.uk short names); NAME_MAP_SALARY bridges them.
transfermarkt_squad_values_adjusted.csv already matches raw_results naming.

Inputs:  raw_results table (infra/data/db/priors.db), prem_salaries_adjusted.csv,
         transfermarkt_squad_values_adjusted.csv
Output:  printed regression summaries only (no files written).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DB_PATH = Path(__file__).resolve().parents[7] / "infra" / "data" / "db" / "priors.db"
SALARY_PATH = Path(__file__).parent / "prem_salaries_adjusted.csv"
VALUE_PATH = Path(__file__).parent / "transfermarkt_squad_values_adjusted.csv"
CURRENT_SEASON = "2026-2027"  # not yet played -- no PL goals exist to validate against

# prem_salaries_adjusted.csv team name -> raw_results PL team name
NAME_MAP_SALARY = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    "Nottingham Forest": "Nott'm Forest",
    "Stoke City": "Stoke",
    "West Bromwich": "West Brom",
    "Wolverhampton": "Wolves",
}


def pl_team_season_goals(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team): games, gf, ga, gf_rate, ga_rate, for PL only."""
    pl = raw[raw["division"] == "Premier_League"]
    home = pl.rename(columns={"home_team": "team", "home_goals": "gf", "away_goals": "ga"})
    away = pl.rename(columns={"away_team": "team", "away_goals": "gf", "home_goals": "ga"})
    long = pd.concat([
        home[["season", "team", "gf", "ga"]],
        away[["season", "team", "gf", "ga"]],
    ], ignore_index=True)

    agg = long.groupby(["season", "team"]).agg(
        games=("gf", "size"), gf=("gf", "sum"), ga=("ga", "sum"),
    ).reset_index()
    agg["gf_rate"] = agg["gf"] / agg["games"]
    agg["ga_rate"] = agg["ga"] / agg["games"]
    return agg


def build_salary_panel(salaries: pd.DataFrame, pl_goals: pd.DataFrame) -> pd.DataFrame:
    df = salaries.copy()
    df["team"] = df["team_name"].replace(NAME_MAP_SALARY)
    merged = df.merge(pl_goals, on=["season", "team"], how="left")
    missing = merged[merged["gf_rate"].isna()]
    if len(missing):
        raise ValueError(f"No PL goals data for: {missing[['team', 'season']].to_dict('records')}")
    merged["log_wage"] = np.log(merged["gross_2025_26_terms"])
    return merged[["team", "season", "gross_2025_26_terms", "log_wage", "gf_rate", "ga_rate"]]


def build_value_panel(values: pd.DataFrame, pl_goals: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season): CPI-adjusted squad market value + PL goals
    rates. `values` team names already match raw_results convention
    (transfermarkt_squad_values was built directly from it), so unlike
    build_salary_panel this needs no name-mapping bridge."""
    merged = values.merge(pl_goals, on=["season", "team"], how="left")
    missing = merged[merged["gf_rate"].isna()]
    if len(missing):
        raise ValueError(f"No PL goals data for: {missing[['team', 'season']].to_dict('records')}")
    merged["log_value"] = np.log(merged["squad_value_2025_26_terms"])
    return merged[["team", "season", "squad_value_2025_26_terms", "log_value", "gf_rate", "ga_rate"]]


def fit_ols(df: pd.DataFrame, x_cols: list[str], y_col: str, label: str):
    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    model = sm.OLS(y, X).fit()
    print(f"\n{label}: {y_col} ~ {' + '.join(x_cols)}")
    print(f"  n={len(df)}, R^2={model.rsquared:.3f}")
    for name, coef, se, p in zip(model.params.index, model.params, model.bse, model.pvalues):
        print(f"  {name:>12}: {coef: .4f}  (se={se:.4f}, p={p:.3f})")
    return model


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    raw = pd.read_sql("SELECT * FROM raw_results", conn)
    conn.close()
    pl_goals = pl_team_season_goals(raw)

    salaries = pd.read_csv(SALARY_PATH)
    salary_panel = build_salary_panel(salaries, pl_goals)
    fit_ols(salary_panel, ["log_wage"], "gf_rate", "WAGE BILL ATTACK")
    fit_ols(salary_panel, ["log_wage"], "ga_rate", "WAGE BILL DEFENSE")

    # exclude the current season: no historical PL goals exist yet to merge against
    values = pd.read_csv(VALUE_PATH)
    values = values[values["season"] != CURRENT_SEASON]
    value_panel = build_value_panel(values, pl_goals)
    fit_ols(value_panel, ["log_value"], "gf_rate", "SQUAD VALUE ATTACK")
    fit_ols(value_panel, ["log_value"], "ga_rate", "SQUAD VALUE DEFENSE")
