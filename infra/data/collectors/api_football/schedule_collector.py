"""
API-Football schedule collector.

Fetches full fixture lists (all competitions) for every Premier League team
across multiple seasons. One request per team-season, so 20 teams × N seasons
+ 1 team-list request = well within the 100 req/day free limit.

Usage:
    export API_FOOTBALL_KEY=your_key_here
    python schedule_collector.py

Outputs:
    infra/data/json/api_football/fixtures_raw/<team_id>_<season>.json  — raw API response
    infra/data/json/api_football/fixtures_flat.csv                     — flattened for analysis
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
PL_LEAGUE_ID = 39        # Premier League
SEASONS = [2021, 2022, 2023, 2024, 2025]   # 2021 = 2021-22, etc.

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "json" / "api_football" / "fixtures_raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLAT_CSV = OUTPUT_DIR.parent / "fixtures_flat.csv"

# Pause between requests to be polite (free tier has no burst limit stated,
# but 1 req/s is safe)
REQUEST_DELAY = 1.2  # seconds


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    if not API_KEY:
        raise RuntimeError("Set API_FOOTBALL_KEY environment variable before running.")
    return {
        "x-apisports-key": API_KEY,
    }


def _get(endpoint: str, params: dict) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
    print(f"  [quota remaining: {remaining}]")
    return data


# ---------------------------------------------------------------------------
# Step 1: get current PL team IDs
# ---------------------------------------------------------------------------

def fetch_pl_teams(season: int = 2024) -> list[dict]:
    """Return list of {id, name} dicts for all teams in the PL that season."""
    cache_path = OUTPUT_DIR.parent / f"pl_teams_{season}.json"
    if cache_path.exists():
        print(f"Using cached team list: {cache_path}")
        return json.loads(cache_path.read_text())

    print(f"Fetching PL team list for {season}...")
    time.sleep(REQUEST_DELAY)
    data = _get("teams", {"league": PL_LEAGUE_ID, "season": season})
    teams = [
        {"id": t["team"]["id"], "name": t["team"]["name"]}
        for t in data.get("response", [])
    ]
    cache_path.write_text(json.dumps(teams, indent=2))
    print(f"  Found {len(teams)} teams.")
    return teams


# ---------------------------------------------------------------------------
# Step 2: fetch fixtures for each team × season
# ---------------------------------------------------------------------------

def fetch_team_season_fixtures(team_id: int, season: int) -> list[dict]:
    """
    Fetch all fixtures for a team in a given season (all competitions).
    Returns the raw 'response' list from the API.
    Uses a local cache so re-running never burns quota.
    """
    cache_path = OUTPUT_DIR / f"{team_id}_{season}.json"
    if cache_path.exists():
        print(f"  Cache hit: {cache_path.name}")
        return json.loads(cache_path.read_text())

    print(f"  Fetching team {team_id}, season {season}...")
    time.sleep(REQUEST_DELAY)
    data = _get("fixtures", {"team": team_id, "season": season})
    fixtures = data.get("response", [])
    cache_path.write_text(json.dumps(fixtures, indent=2))
    print(f"    → {len(fixtures)} fixtures")
    return fixtures


# ---------------------------------------------------------------------------
# Step 3: flatten to CSV
# ---------------------------------------------------------------------------

def flatten_fixtures(all_fixtures: list[dict]) -> pd.DataFrame:
    rows = []
    for f in all_fixtures:
        fixture = f.get("fixture", {})
        league  = f.get("league", {})
        teams   = f.get("teams", {})
        goals   = f.get("goals", {})
        score   = f.get("score", {})

        rows.append({
            "fixture_id":       fixture.get("id"),
            "date":             fixture.get("date"),
            "timestamp":        fixture.get("timestamp"),
            "status":           fixture.get("status", {}).get("short"),
            "venue":            fixture.get("venue", {}).get("name"),
            "league_id":        league.get("id"),
            "league_name":      league.get("name"),
            "league_country":   league.get("country"),
            "season":           league.get("season"),
            "round":            league.get("round"),
            "home_team_id":     teams.get("home", {}).get("id"),
            "home_team":        teams.get("home", {}).get("name"),
            "away_team_id":     teams.get("away", {}).get("id"),
            "away_team":        teams.get("away", {}).get("name"),
            "home_goals":       goals.get("home"),
            "away_goals":       goals.get("away"),
            "ht_home":          score.get("halftime", {}).get("home"),
            "ht_away":          score.get("halftime", {}).get("away"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/London")
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    teams = fetch_pl_teams(season=2025)

    # Collect every team across all seasons — teams promoted/relegated won't
    # have PL fixtures in off-seasons but their cup/European data still lands.
    all_fixtures: list[dict] = []

    total = len(teams) * len(SEASONS)
    done  = 0
    for team in teams:
        for season in SEASONS:
            done += 1
            print(f"[{done}/{total}] {team['name']} — {season}")
            fixtures = fetch_team_season_fixtures(team["id"], season)
            all_fixtures.extend(fixtures)

    # Deduplicate (same fixture appears for both home and away team)
    seen: set[int] = set()
    unique: list[dict] = []
    for f in all_fixtures:
        fid = f.get("fixture", {}).get("id")
        if fid not in seen:
            seen.add(fid)
            unique.append(f)

    print(f"\nTotal unique fixtures: {len(unique)}")

    df = flatten_fixtures(unique)
    df.to_csv(FLAT_CSV, index=False)
    print(f"Saved flat CSV → {FLAT_CSV}")
    print(df["league_name"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
