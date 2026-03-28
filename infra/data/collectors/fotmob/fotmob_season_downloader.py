"""
FotMob season downloader.

Finds missing match IDs automatically, opens all URLs in your real browser
at once, then walks you through saving each one from clipboard.
"""

import json
import subprocess
import webbrowser
from pathlib import Path

from playwright.sync_api import sync_playwright

LEAGUES = [
    {"id": 47, "name": "Premier_League", "season_type": "winter"},
    {"id": 46, "name": "Superligaen", "season_type": "winter"},
    #{"id": 48, "name": "Championship", "season_type": "winter"},
    #{"id": 87, "name": "La_Liga", "season_type": "winter"},
    #{"id": 59, "name": "Eliteserien", "season_type": "summer"},
    #{"id": 67, "name": "Allsvenskan", "season_type": "summer"},
    #{"id": 126, "name": "League_of_Ireland", "season_type": "summer"},
]

FIREFOX_PROFILE = str(Path.home() / "Library/Application Support/Firefox/Profiles/zx2gcfse.default-release")
BASE_URL = "https://www.fotmob.com/api/data"


def _get_missing_ids(league, season_str, existing):
    """Use Playwright Firefox with real profile to fetch the fixtures list."""
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    options = Options()
    options.add_argument("-profile")
    options.add_argument(FIREFOX_PROFILE)
    options.add_argument("--headless")

    driver = webdriver.Firefox(options=options)
    try:
        driver.set_script_timeout(15)
        driver.get("https://www.fotmob.com")

        result = driver.execute_async_script(f"""
            const callback = arguments[arguments.length - 1];
            fetch('/api/data/fixtures?id={league["id"]}&season={season_str}')
                .then(r => r.json())
                .then(data => callback({{ok: true, data: data}}))
                .catch(err => callback({{ok: false, error: err.toString()}}));
        """)

        if not result or not result.get("ok"):
            raise Exception(result.get("error") if result else "No response")

        fixtures = result["data"]
        match_ids = [
            str(x["id"]) for x in fixtures
            if not x["status"]["cancelled"] and x["status"]["finished"]
        ]
        return match_ids, [m for m in match_ids if m not in existing]
    finally:
        driver.quit()


def _save_from_clipboard(match_id, json_dir):
    """Read JSON from clipboard and save to file."""
    raw = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout.strip()
    if not raw:
        print(f"    [{match_id}] Clipboard empty, skipping.")
        return False
    try:
        data = json.loads(raw)
        if "error" in data:
            print(f"    [{match_id}] Got error response: {data}, skipping.")
            return False
        out_path = json_dir / f"{match_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"    [{match_id}] Saved.")
        return True
    except json.JSONDecodeError as e:
        print(f"    [{match_id}] Invalid JSON: {e}")
        return False


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
    print(f"    Fetching fixtures list...")

    match_ids, to_get = _get_missing_ids(league, season_str, existing)

    print(f"    {len(match_ids)} valid matches")
    if not to_get:
        print("    Nothing new to fetch.\n")
        return

    print(f"\n    {len(to_get)} missing match{'es' if len(to_get) > 1 else ''}.")
    print("    Opening all URLs in your browser now...")

    json_dir.mkdir(parents=True, exist_ok=True)
    urls = [f"{BASE_URL}/matchDetails?matchId={m}" for m in to_get]
    for url in urls:
        webbrowser.open(url)

    print(f"\n    For each tab: Cmd+A, Cmd+C, then press Enter here.")
    print("    (If Cloudflare blocks a tab, skip it with just Enter.)\n")

    for match_id in to_get:
        input(f"    [{match_id}] Ready (Cmd+A, Cmd+C done)? Press Enter... ")
        _save_from_clipboard(match_id, json_dir)

    print("\nDone\n")


def main():
    for league in LEAGUES:
        for season_start in range(2025, 2026):
            store_season(league, season_start)


if __name__ == "__main__":
    main()
