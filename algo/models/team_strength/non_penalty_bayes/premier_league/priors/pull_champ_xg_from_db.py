"""
Derives Championship team-season totals (non-penalty xG, non-penalty xGA,
non-penalty goals, non-penalty goals against, possession) from the local
fotmob.db, replacing the FotMob-public-API version of champ_xg_history.csv.

Same sourcing pattern as pull_prem_xg_from_db.py:
  - np_matches for goals (penalty-stripped by fotmob_etl_database_non_penalty.py)
  - match_stats.home/away_expected_goals_non_penalty for xG
  - match_stats.home/away_BallPossesion for possession
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("infra/data/db/fotmob.db")
OUTPUT_CSV = Path(__file__).parent / "champ_xg_history.csv"

QUERY = """
SELECT
    m.season,
    m.home_team AS team_id,
    s.home_expected_goals_non_penalty AS xg_for,
    s.away_expected_goals_non_penalty AS xg_against,
    m.home_goals AS goals_for,
    m.away_goals AS goals_against,
    s.home_BallPossesion AS possession
FROM np_matches m
JOIN match_stats s ON m.match_id = s.match_id
WHERE m.league_id = 'Championship'

UNION ALL

SELECT
    m.season,
    m.away_team AS team_id,
    s.away_expected_goals_non_penalty AS xg_for,
    s.home_expected_goals_non_penalty AS xg_against,
    m.away_goals AS goals_for,
    m.home_goals AS goals_against,
    s.away_BallPossesion AS possession
FROM np_matches m
JOIN match_stats s ON m.match_id = s.match_id
WHERE m.league_id = 'Championship'
"""


def main():
    conn = sqlite3.connect(DB_PATH)
    matches = pd.read_sql(QUERY, conn)
    team_names = pd.read_sql("SELECT team_id, team_name FROM team_id_mapping", conn)
    conn.close()

    matches["xg_for"] = matches["xg_for"].astype(float)
    matches["xg_against"] = matches["xg_against"].astype(float)

    season_totals = (
        matches.groupby(["season", "team_id"])
        .agg(
            matches_played=("goals_for", "size"),
            xg=("xg_for", "sum"),
            goals=("goals_for", "sum"),
            xga=("xg_against", "sum"),
            goals_against=("goals_against", "sum"),
            possession=("possession", "mean"),
        )
        .reset_index()
    )

    season_totals["season"] = season_totals["season"].str.replace("-", "/", regex=False)
    season_totals = season_totals.merge(team_names, on="team_id").rename(columns={"team_name": "team"})
    season_totals = season_totals[
        ["season", "team_id", "team", "matches_played", "xg", "goals", "xga", "goals_against", "possession"]
    ]
    season_totals = season_totals.sort_values(["season", "xg"], ascending=[True, False])
    season_totals["possession"] = season_totals["possession"].round(1)
    season_totals[["xg", "xga"]] = season_totals[["xg", "xga"]].round(2)

    season_totals.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(season_totals)} team-seasons across {season_totals['season'].nunique()} seasons to {OUTPUT_CSV}")
    print(sorted(season_totals["season"].unique()))


if __name__ == "__main__":
    main()
