"""
Downloads historical match results (final scores only, no odds/shots) for the
Premier League (E0) and Championship (E1) from football-data.co.uk, going back
to 1993-94. Used to build the Championship -> PL goal-rate translation used for
cold-start priors on newly promoted teams. Goes back further than fotmob.db
(which only covers 2020-21 onward), so this is the source for the long-run
promotion/wage regressions in algo/models/.../premier_league/priors/.

Output: raw_results table in infra/data/db/priors.db, columns
    division, season, home_team, away_team, home_goals, away_goals
where season is formatted "YYYY-YYYY" (e.g. "2023-2024").

Usage
-----
    python fetch_historical_results.py
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests

DB_PATH = Path("infra/data/db/priors.db")
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
DIVISIONS = {"E0": "Premier_League", "E1": "Championship"}
FIRST_SEASON_START = 1993   # 1993-94 is football-data's earliest coverage
LAST_SEASON_START = 2025    # 2025-26, the most recently completed season


def _season_code(start_year: int) -> str:
    """1993 -> '9394', 2023 -> '2324'."""
    a = start_year % 100
    b = (start_year + 1) % 100
    return f"{a:02d}{b:02d}"


def _season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def fetch_all() -> pd.DataFrame:
    rows = []
    for start_year in range(FIRST_SEASON_START, LAST_SEASON_START + 1):
        code = _season_code(start_year)
        for div, div_name in DIVISIONS.items():
            url = BASE_URL.format(code=code, div=div)
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  [{div_name} {_season_label(start_year)}] skip: {e}")
                continue

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), usecols=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            if df.empty:
                print(f"  [{div_name} {_season_label(start_year)}] empty, skip")
                continue

            df["division"] = div_name
            df["season"] = _season_label(start_year)
            df = df.rename(columns={
                "HomeTeam": "home_team", "AwayTeam": "away_team",
                "FTHG": "home_goals", "FTAG": "away_goals",
            })
            rows.append(df[["division", "season", "home_team", "away_team", "home_goals", "away_goals"]])
            print(f"  [{div_name} {_season_label(start_year)}] {len(df)} matches")
            time.sleep(0.2)  # be polite

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    all_results = fetch_all()
    conn = sqlite3.connect(DB_PATH)
    all_results.to_sql("raw_results", conn, if_exists="replace", index=False)
    conn.close()
    print(f"\nSaved {len(all_results)} matches -> {DB_PATH}::raw_results")
