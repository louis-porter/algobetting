"""
Downloads total squad market value per club per season from Transfermarkt,
covering the same team-seasons present in raw_results.csv (Premier League,
2015-16 onward). Squad value is summed from individual player market values
on each club's "detailed squad" page for that season (saison_id = start year),
which reflects Transfermarkt's valuation as of that season's page snapshot
(effectively partway through the season, same convention TM uses site-wide).

Team name -> Transfermarkt club ID is resolved once via TM's search endpoint
and cached to transfermarkt_team_ids.json (the URL slug itself doesn't need
to match TM's canonical slug — only the numeric verein ID matters).

Output: transfermarkt_squad_values.csv with columns
    team, season, squad_value_eur, n_players
where team matches raw_results.csv naming and season is "YYYY-YYYY".

Usage
-----
    python pull_transfermarkt_values.py
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "raw_results.csv"
ID_CACHE_PATH = HERE / "transfermarkt_team_ids.json"
OUT_PATH = HERE / "transfermarkt_squad_values.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
SLEEP = 1.2
FIRST_SEASON_START = 2015

# raw_results.csv team name -> Transfermarkt search query
TM_QUERY = {
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford FC",
    "Brighton": "Brighton and Hove Albion",
    "Burnley": "Burnley FC",
    "Cardiff": "Cardiff City",
    "Chelsea": "Chelsea FC",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Huddersfield": "Huddersfield Town",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Liverpool": "Liverpool FC",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Middlesbrough": "Middlesbrough FC",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "FC Southampton",
    "Stoke": "Stoke City",
    "Sunderland": "AFC Sunderland",
    "Swansea": "Swansea City",
    "Tottenham": "Tottenham Hotspur",
    "Watford": "Watford FC",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}

_YOUTH_RE = re.compile(r"-u1[6-9]$|-u2[0-3]$|-jugend$|-frauen$|-women$|-ii$")


def _get(url: str) -> str | None:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print(f"    retry ({attempt+1}/3) {url}: {e}")
            time.sleep(2)
    return None


def resolve_team_id(query: str) -> int | None:
    url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={query.replace(' ', '+')}"
    html = _get(url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=re.compile(r"/startseite/verein/\d+")):
        slug = a["href"].split("/")[1]
        text = a.get_text(strip=True)
        if not text or slug in ("-tm", "unbekannt") or _YOUTH_RE.search(slug):
            continue
        return int(re.search(r"verein/(\d+)", a["href"]).group(1))
    return None


def load_team_ids(teams: list[str]) -> dict[str, int]:
    cache = json.loads(ID_CACHE_PATH.read_text()) if ID_CACHE_PATH.exists() else {}
    for team in teams:
        if team in cache:
            continue
        query = TM_QUERY[team]
        team_id = resolve_team_id(query)
        print(f"  resolved {team!r} ({query}) -> {team_id}")
        cache[team] = team_id
        ID_CACHE_PATH.write_text(json.dumps(cache, indent=2))
        time.sleep(SLEEP)
    return cache


def _parse_value(text: str) -> float | None:
    text = text.strip().replace("€", "")
    if not text or text == "-":
        return None
    mult = 1.0
    if text.endswith("bn"):
        mult, text = 1e9, text[:-2]
    elif text.endswith("m"):
        mult, text = 1e6, text[:-1]
    elif text.endswith("k"):
        mult, text = 1e3, text[:-1]
    try:
        return float(text) * mult
    except ValueError:
        return None


def fetch_squad_value(team_id: int, season_start: int) -> tuple[float, int] | None:
    url = f"https://www.transfermarkt.com/club/kader/verein/{team_id}/plus/0?saison_id={season_start}"
    html = _get(url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="items")
    if table is None or table.find("tbody") is None:
        return None
    total, n = 0.0, 0
    for row in table.find("tbody").find_all("tr", recursive=False):
        cell = row.find("td", class_=lambda c: c and "rechts" in c.split() and "hauptlink" in c.split())
        if cell is None:
            continue
        value = _parse_value(cell.get_text(strip=True))
        if value is not None:
            total += value
            n += 1
    return (total, n) if n else None


def build_team_season_list() -> pd.DataFrame:
    raw = pd.read_csv(RESULTS_PATH)
    pl = raw[(raw["division"] == "Premier_League")
             & (raw["season"] >= f"{FIRST_SEASON_START}-{FIRST_SEASON_START + 1}")]
    home = pl[["season", "home_team"]].rename(columns={"home_team": "team"})
    away = pl[["season", "away_team"]].rename(columns={"away_team": "team"})
    return pd.concat([home, away], ignore_index=True).drop_duplicates().sort_values(["team", "season"])


if __name__ == "__main__":
    team_seasons = build_team_season_list()
    teams = sorted(team_seasons["team"].unique())
    print(f"{len(teams)} teams, {len(team_seasons)} team-seasons to fetch\n")

    print("Resolving team IDs...")
    team_ids = load_team_ids(teams)
    missing = [t for t in teams if team_ids.get(t) is None]
    if missing:
        print(f"WARNING: could not resolve IDs for: {missing}")

    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame(columns=["team", "season", "squad_value_eur", "n_players"])
    done = set(zip(existing["team"], existing["season"]))

    rows = [existing] if len(existing) else []
    print("\nFetching squad values...")
    for _, r in team_seasons.iterrows():
        team, season = r["team"], r["season"]
        if (team, season) in done:
            continue
        team_id = team_ids.get(team)
        if team_id is None:
            continue
        season_start = int(season.split("-")[0])
        result = fetch_squad_value(team_id, season_start)
        if result is None:
            print(f"  [{team} {season}] FAILED")
            continue
        value, n = result
        print(f"  [{team} {season}] EUR {value/1e6:.1f}m ({n} players)")
        rows.append(pd.DataFrame([{"team": team, "season": season, "squad_value_eur": value, "n_players": n}]))
        pd.concat(rows, ignore_index=True).to_csv(OUT_PATH, index=False)
        time.sleep(SLEEP)

    final = pd.concat(rows, ignore_index=True) if rows else existing
    final.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(final)} team-seasons -> {OUT_PATH}")
