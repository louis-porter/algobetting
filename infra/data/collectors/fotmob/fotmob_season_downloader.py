import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

from curl_cffi import requests

LEAGUES = [
    {"id": 47, "name": "Premier_League", "season_type": "winter"},
    {"id": 46, "name": "Superligaen", "season_type": "winter"}
    #{"id": 48, "name": "Championship", "season_type": "winter"},
    #{"id": 87, "name": "La_Liga", "season_type": "winter"},
    #{"id": 59, "name": "Eliteserien", "season_type": "summer"},
    # {"id": 67, "name": "Allsvenskan", "season_type": "summer"},
    # {"id": 126, "name": "League_of_Ireland", "season_type": "summer"},
]
BASE_URL = "https://www.fotmob.com/api/data"

# CHANGE THIS HEADER
#HEADERS = {
#    "x-mas": "eyJib2R5Ijp7InVybCI6Ii9hcGkvZGF0YS9tYXRjaGVzP2RhdGU9MjAyNTEwMDgmdGltZXpvbmU9RXVyb3BlJTJGTG9uZG9uJmNjb2RlMz1HQlIiLCJjb2RlIjoxNzU5OTE5ODY3NjQ3LCJmb28iOiJwcm9kdWN0aW9uOjBkYzg4ZDUyM2U2Y2Y3OWZlYzNiNzUwZGFhNDgwODkyOGYwNDliMWYifSwic2lnbmF0dXJlIjoiMzM3MkUwM0E3Q0ZDQkE0NzAxOTcyNzA2QUYwQTlGMDMifQ=="
#}

def fetch_match(session, match_id):
    url = f"{BASE_URL}/matchDetails?matchId={match_id}"
    r = session.get(url)
    r.raise_for_status()
    return r.json()["content"]


def save_match(match_data, league_folder, season_folder, match_id):
    folder = Path(f"infra/data/json/{league_folder}/{season_folder}")
    folder.mkdir(parents=True, exist_ok=True)
    filepath = folder / f"{match_id}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(match_data, f, ensure_ascii=False)


def fetch_and_save_match(session, league_folder, season_folder, match_id):
    try:
        match_data = fetch_match(session, match_id)
        save_match(match_data, league_folder, season_folder, match_id)
        return match_id, True

    except Exception as e:
        return match_id, str(e)


def store_season(session, league, season_start):
    if league["season_type"] == "summer":
        season_str = str(season_start)
    else:
        season_str = f"{season_start}/{season_start + 1}"

    season_folder = season_str.replace("/", "-")
    folder = f"data/{league['name']}/{season_folder}"

    existing = glob(f"{folder}/*.json")  # find existing files for given league-season pair
    existing = [x.split("/")[-1].split(".")[0] for x in existing]

    url = f"{BASE_URL}/fixtures?id={league['id']}&season={season_str}"
    r = session.get(url)
    match_ids = [str(x["id"]) for x in r.json() if not x["status"]["cancelled"] and x["status"]["finished"]]
    to_get = list(set(match_ids) - set(existing))

    print(f"{league['name']} {season_str}")
    print(f"    {len(existing)} matches already stored")
    print(f"    {len(match_ids)} valid matches")
    if len(to_get) == 0:
        print()
        return
    print(f"    Fetching {len(to_get)} match{'es' if len(to_get) > 1 else ''}...")

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(fetch_and_save_match, session, league["name"], season_folder, match_id) for match_id in to_get]

        for future in as_completed(futures):
            match_id, result = future.result()
            if result is not True:
                print(f"Error fetching match {match_id}: {result}")

    print("Done\n")
    return


def main():
    xmas = input("Paste in xmas: ")
    HEADERS = {
        "x-mas": xmas
    }
    with requests.Session() as session:
        session.headers.update(HEADERS)
        for league in LEAGUES:
            for season_start in range(2025, 2026):
                store_season(session, league, season_start)


if __name__ == "__main__":
    main()
