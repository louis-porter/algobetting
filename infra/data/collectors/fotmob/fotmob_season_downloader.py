"""
FotMob season downloader.

Finds missing match IDs via the fixtures endpoint and fetches matchDetails
for each directly over HTTP - no manual copy/paste, no browser needed.
(The old workflow assumed matchDetails was Cloudflare-gated; as of writing,
plain requests with a browser User-Agent goes straight through on both the
fixtures and matchDetails endpoints.)
"""

import json
import time
from pathlib import Path

import requests

LEAGUES = [
    {"id": 47, "name": "Premier_League", "season_type": "winter"},
    {"id": 46, "name": "Superligaen", "season_type": "winter"},
    #{"id": 48, "name": "Championship", "season_type": "winter"},
    #{"id": 87, "name": "La_Liga", "season_type": "winter"},
    #{"id": 59, "name": "Eliteserien", "season_type": "summer"},
    #{"id": 67, "name": "Allsvenskan", "season_type": "summer"},
    #{"id": 126, "name": "League_of_Ireland", "season_type": "summer"},
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
BASE_URL = "https://www.fotmob.com/api/data"
REQUEST_DELAY = 1.5
MAX_RETRIES = 3


def _get_finished_match_ids(league, season_str):
    resp = requests.get(
        f"{BASE_URL}/fixtures",
        params={"id": league["id"], "season": season_str},
        headers=HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    fixtures = resp.json()
    return [
        str(f["id"])
        for f in fixtures
        if f["status"]["finished"] and not f["status"]["cancelled"]
    ]


def _fetch_match_details(match_id, session):
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(f"{BASE_URL}/matchDetails", params={"matchId": match_id}, timeout=20)
        try:
            data = resp.json()
        except ValueError:
            data = None

        if resp.status_code == 200 and data and "general" in data:
            return data

        if attempt < MAX_RETRIES:
            wait_s = 5 * attempt
            print(f"    [{match_id}] attempt {attempt} failed (status {resp.status_code}), retrying in {wait_s}s...")
            time.sleep(wait_s)

    return None


def store_season(league, season_start):
    if league["season_type"] == "summer":
        season_str = str(season_start)
    else:
        season_str = f"{season_start}/{season_start + 1}"

    season_folder = season_str.replace("/", "-")
    json_dir = Path(f"infra/data/json/{league['name']}/{season_folder}")
    existing = {p.stem for p in json_dir.glob("*.json")} if json_dir.exists() else set()

    print(f"{league['name']} {season_str}")
    print(f"    {len(existing)} matches already stored")
    print("    Fetching fixtures list...")

    match_ids = _get_finished_match_ids(league, season_str)
    to_get = [m for m in match_ids if m not in existing]

    print(f"    {len(match_ids)} valid matches")
    if not to_get:
        print("    Nothing new to fetch.\n")
        return

    print(f"    {len(to_get)} missing match{'es' if len(to_get) > 1 else ''}. Fetching...")
    json_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    failures = []
    for i, match_id in enumerate(to_get, 1):
        data = _fetch_match_details(match_id, session)
        if data is None:
            print(f"    [{i}/{len(to_get)}] {match_id} FAILED, skipping.")
            failures.append(match_id)
        else:
            out_path = json_dir / f"{match_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"    [{i}/{len(to_get)}] {match_id} saved.")
        time.sleep(REQUEST_DELAY)

    if failures:
        print(f"\n    {len(failures)} failures: {failures}")
    print("\nDone\n")


def main():
    for league in LEAGUES:
        for season_start in range(2025, 2026):
            store_season(league, season_start)


if __name__ == "__main__":
    main()
