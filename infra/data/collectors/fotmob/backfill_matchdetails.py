"""
One-off backfill: fetches raw matchDetails JSON for a season directly via
requests (no Cloudflare challenge encountered on this endpoint currently,
unlike when fotmob_season_downloader.py's manual clipboard workflow was
built), and saves into the same infra/data/json/{league}/{season}/ layout
that fotmob_etl_database.py expects.

Usage:
    python backfill_matchdetails.py <league_id> <season_dash> <match_id...>
    python backfill_matchdetails.py --season <league_numeric_id> <league_id> <season_dash>
        (resolves match IDs itself via the fixtures endpoint)
"""

import json
import sys
import time
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.5
MAX_RETRIES = 5
RETRY_WAIT = 15  # seconds; covers brief network drops (e.g. laptop waking from sleep)


def get_finished_match_ids(league_numeric_id: int, season_slash: str) -> list[int]:
    """season_slash like '2020/2021'. Returns finished, non-cancelled match IDs."""
    resp = requests.get(
        "https://www.fotmob.com/api/data/fixtures",
        params={"id": league_numeric_id, "season": season_slash},
        headers=HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    fixtures = resp.json()
    return [
        int(f["id"])
        for f in fixtures
        if f["status"]["finished"] and not f["status"]["cancelled"]
    ]


def fetch_missing(league_id: str, season: str, match_ids: list[int], db_path="infra/data/db/fotmob.db"):
    json_dir = Path(f"infra/data/json/{league_id}/{season}")
    json_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in json_dir.glob("*.json")}

    to_fetch = [m for m in match_ids if str(m) not in existing]
    print(f"{len(existing)} already saved, {len(to_fetch)} to fetch.")

    session = requests.Session()
    session.headers.update(HEADERS)

    failures = []
    for i, match_id in enumerate(to_fetch, 1):
        data = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = session.get(
                    "https://www.fotmob.com/api/data/matchDetails",
                    params={"matchId": match_id},
                    timeout=20,
                )
                candidate = resp.json()
                if resp.status_code == 200 and "general" in candidate:
                    data = candidate
                    break
                print(f"  [{i}/{len(to_fetch)}] {match_id} attempt {attempt} bad response (status {resp.status_code})")
            except requests.exceptions.RequestException as e:
                # Covers dropped/no connection - e.g. laptop asleep or waking up.
                print(f"  [{i}/{len(to_fetch)}] {match_id} attempt {attempt} network error: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)

        if data is not None:
            (json_dir / f"{match_id}.json").write_text(json.dumps(data))
            print(f"  [{i}/{len(to_fetch)}] {match_id} saved")
        else:
            print(f"  [{i}/{len(to_fetch)}] {match_id} FAILED after {MAX_RETRIES} attempts, skipping.")
            failures.append(match_id)

        time.sleep(REQUEST_DELAY)

    if failures:
        print(f"\n{len(failures)} failures: {failures}")
    return failures


if __name__ == "__main__":
    league_id, season = sys.argv[1], sys.argv[2]
    ids = [int(x) for x in sys.argv[3:]]
    fetch_missing(league_id, season, ids)
