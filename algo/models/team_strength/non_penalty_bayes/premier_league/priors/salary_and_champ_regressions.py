"""
Two pairs of simple OLS regressions:

1. Championship form -> promoted team's first-PL-season scoring:
     pl_gf_rate ~ champ_xG + champ_pos          (ATTACK)
     pl_ga_rate ~ champ_xGA + champ_pos         (DEFENSE)
   Uses champ_performance.csv (final Championship season xG/xGA/possession
   for each promoted team) joined to that team's first PL season goals
   rate from raw_results.csv.

2. Wage bill -> same-season PL scoring, across all PL team-seasons:
     pl_gf_rate ~ log(gross_2025_26_terms)      (ATTACK)
     pl_ga_rate ~ log(gross_2025_26_terms)      (DEFENSE)
   Uses prem_salaries_adjusted.csv (CPI-adjusted wage bill, see
   adjust_salaries_for_inflation.py) joined to that team's own-season PL
   goals rate from raw_results.csv. Wage is log-transformed since spending
   spans two orders of magnitude and marginal returns on wages taper off.

Team names differ across the three source files (raw_results.csv uses
football-data.co.uk short names); NAME_MAP_* below bridge them.

Inputs:  raw_results.csv, champ_performance.csv, prem_salaries_adjusted.csv
Output:  printed regression summaries only (no files written).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

RESULTS_PATH = Path(__file__).parent / "raw_results.csv"
CHAMP_PERF_PATH = Path(__file__).parent / "champ_performance.csv"
SALARY_PATH = Path(__file__).parent / "prem_salaries_adjusted.csv"
VALUE_PATH = Path(__file__).parent / "transfermarkt_squad_values_adjusted.csv"

# champ_performance.csv team name -> raw_results.csv PL team name
NAME_MAP_CHAMP = {
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
}

# prem_salaries_adjusted.csv team name -> raw_results.csv PL team name
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


def next_season(season: str) -> str:
    start = int(season.split("-")[0])
    return f"{start + 1}-{start + 2}"


def build_champ_panel(champ_perf: pd.DataFrame, pl_goals: pd.DataFrame) -> pd.DataFrame:
    df = champ_perf.rename(columns={"Year of Promotion": "champ_season", "Team": "team", "Pos": "pos"})
    df["team"] = df["team"].replace(NAME_MAP_CHAMP)
    df["pl_season"] = df["champ_season"].apply(next_season)

    merged = df.merge(
        pl_goals, left_on=["pl_season", "team"], right_on=["season", "team"], how="left"
    )
    missing = merged[merged["gf_rate"].isna()]
    if len(missing):
        raise ValueError(f"No PL goals data for: {missing[['team', 'pl_season']].to_dict('records')}")
    return merged[["team", "champ_season", "pl_season", "xG", "xGA", "pos", "gf_rate", "ga_rate"]]


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
    rates. `values` team names already match raw_results.csv convention
    (transfermarkt_squad_values.csv was built directly from it), so unlike
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
    raw = pd.read_csv(RESULTS_PATH)
    pl_goals = pl_team_season_goals(raw)

    champ_perf = pd.read_csv(CHAMP_PERF_PATH)
    champ_panel = build_champ_panel(champ_perf, pl_goals)
    fit_ols(champ_panel, ["xG", "pos"], "gf_rate", "CHAMPIONSHIP ATTACK")
    fit_ols(champ_panel, ["xGA", "pos"], "ga_rate", "CHAMPIONSHIP DEFENSE")

    salaries = pd.read_csv(SALARY_PATH)
    salary_panel = build_salary_panel(salaries, pl_goals)
    fit_ols(salary_panel, ["log_wage"], "gf_rate", "WAGE BILL ATTACK")
    fit_ols(salary_panel, ["log_wage"], "ga_rate", "WAGE BILL DEFENSE")

    values = pd.read_csv(VALUE_PATH)
    value_panel = build_value_panel(values, pl_goals)
    fit_ols(value_panel, ["log_value"], "gf_rate", "SQUAD VALUE ATTACK")
    fit_ols(value_panel, ["log_value"], "ga_rate", "SQUAD VALUE DEFENSE")

    # Does Championship form add anything for promoted teams once their
    # (incoming PL season) wage bill is already known?
    combined = champ_panel.merge(
        salary_panel[["team", "season", "log_wage"]],
        left_on=["team", "pl_season"], right_on=["team", "season"], how="left",
    )
    missing = combined[combined["log_wage"].isna()]
    if len(missing):
        raise ValueError(f"No wage data for: {missing[['team', 'pl_season']].to_dict('records')}")
    fit_ols(combined, ["log_wage", "xG", "pos"], "gf_rate", "COMBINED ATTACK (promoted teams)")
    fit_ols(combined, ["log_wage", "xGA", "pos"], "ga_rate", "COMBINED DEFENSE (promoted teams)")
