"""
FotMob Championship season-stats scraper.

Pulls end-of-season team totals for the EFL Championship (xG, xGA, goals,
goals against, possession) as far back as FotMob's advanced-stats coverage
goes (2016/17 — earlier seasons have no xG on FotMob).

Two-step pull:
  1. `leagues?id=48&season=<season>` to resolve each season's numeric
     season ID (via the `fetchAllUrl` on the season's top-stat widget).
  2. `data.fotmob.com/stats/48/season/<id>/<stat>.json` for the actual
     team lists (a plain CDN endpoint - no Cloudflare challenge, unlike
     FotMob's matchDetails endpoint used elsewhere in this repo).

Raw JSON per season/stat is cached under infra/data/json/Championship/season_stats/
so re-runs don't re-hit the network. The merged season table is written to
output/championship_xg_history.csv.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

LEAGUE_ID = 48
LEAGUE_NAME = "Championship"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

STATS = {
    "expected_goals_team": ("xg", "goals"),
    "expected_goals_conceded_team": ("xga", "goals_against"),
    "possession_percentage_team": ("possession", None),
}

JSON_DIR = Path("infra/data/json/Championship/season_stats")
OUTPUT_CSV = Path("output/championship_xg_history.csv")
REQUEST_DELAY = 0.5


def get_available_seasons() -> list[str]:
    resp = requests.get(
        "https://www.fotmob.com/api/data/leagues",
        params={"id": LEAGUE_ID},
        headers=HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["allAvailableSeasons"]


def resolve_season_id(season: str) -> str | None:
    """Returns FotMob's numeric season id, or None if the season has no stats page."""
    resp = requests.get(
        "https://www.fotmob.com/api/data/leagues",
        params={"id": LEAGUE_ID, "season": season},
        headers=HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    players = resp.json().get("stats", {}).get("players", [])
    if not players:
        return None
    match = re.search(r"/season/(\d+)/", players[0].get("fetchAllUrl", ""))
    return match.group(1) if match else None


def fetch_stat(season_id: str, stat_name: str) -> dict | None:
    """Returns the stat payload, or None if FotMob has no data for this
    season/stat combo (xG/xGA weren't tracked for the Championship before
    2018/19, even though the season itself has other stats)."""
    cache_path = JSON_DIR / season_id / f"{stat_name}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return cached if cached else None

    resp = requests.get(
        f"https://data.fotmob.com/stats/{LEAGUE_ID}/season/{season_id}/{stat_name}.json",
        headers=HEADERS,
        timeout=20,
    )
    if resp.status_code in (403, 404):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("null")
        return None
    resp.raise_for_status()
    data = resp.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    time.sleep(REQUEST_DELAY)
    return data


def season_dataframe(season: str, season_id: str) -> pd.DataFrame | None:
    merged = None
    for stat_name, (value_col, sub_col) in STATS.items():
        data = fetch_stat(season_id, stat_name)
        if data is None:
            print(f"  {season}: no '{stat_name}' data, skipping that column.")
            continue
        rows = data["TopLists"][0]["StatList"]
        cols = {
            "TeamId": "team_id",
            "ParticipantName": "team",
            "StatValue": value_col,
            "MatchesPlayed": "matches_played",
        }
        df = pd.DataFrame(rows)[list(cols)].rename(columns=cols)
        if sub_col:
            df[sub_col] = pd.DataFrame(rows)["SubStatValue"]

        if merged is None:
            merged = df
        else:
            merge_cols = ["team_id", "team", "matches_played"]
            merged = merged.merge(df, on=merge_cols, how="outer")

    if merged is None:
        return None
    merged.insert(0, "season", season)
    return merged


def main():
    seasons = get_available_seasons()
    all_rows = []

    for season in seasons:
        season_id = resolve_season_id(season)
        if season_id is None:
            print(f"{season}: no advanced stats on FotMob, stopping backfill here.")
            break
        print(f"{season}: season_id={season_id}")
        df = season_dataframe(season, season_id)
        if df is not None:
            all_rows.append(df)
        time.sleep(REQUEST_DELAY)

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["season", "xg"], ascending=[True, False])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(result)} team-seasons to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
