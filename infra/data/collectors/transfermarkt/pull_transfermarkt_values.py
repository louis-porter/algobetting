"""
Downloads per-player market value + age from each club's "detailed squad"
page on Transfermarkt for that season (saison_id = start year), which
reflects Transfermarkt's valuation as of that season's page snapshot
(effectively partway through the season, same convention TM uses site-wide).
Unvalued players (no market value listed, mostly fringe youth) are dropped --
there's nothing to sum or fit an age curve on for them.

Team-seasons to fetch are discovered from fotmob.db (covers 2020-21 onward).
Seasons from 2015-16 through 2019-20 were backfilled once from raw_results
(football-data.co.uk, dropped as a dependency here since fotmob.db doesn't
go back that far) and stay cached indefinitely -- this function only needs
to know what's missing, not the full history.

Team name -> Transfermarkt club ID is resolved once via TM's search endpoint
and cached to transfermarkt_team_ids.json (the URL slug itself doesn't need
to match TM's canonical slug — only the numeric verein ID matters).

Output: two tables in infra/data/db/priors.db, both keyed by team/season
("YYYY-YYYY"):
    transfermarkt_player_values -- one row per valued player: team, season,
        player, age, value_eur
    transfermarkt_squad_values -- team, season, squad_value_eur, n_players
        (derived by summing transfermarkt_player_values per team-season, kept
        as its own table since it's what the value-regression notebooks read)

Usage
-----
    python pull_transfermarkt_values.py
"""

import json
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).parent
DB_PATH = Path("infra/data/db/priors.db")
FOTMOB_DB_PATH = Path("infra/data/db/fotmob.db")
ID_CACHE_PATH = HERE / "transfermarkt_team_ids.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
SLEEP = 1.2

# fotmob.db team_name -> this script's naming convention (only real mismatches;
# everything else is used as-is)
FOTMOB_TO_TM_NAME = {
    "Nottm Forest": "Nott'm Forest",
    "Luton Town": "Luton",
    "Sheff Utd": "Sheffield United",
}

# team name (this script's convention) -> Transfermarkt search query
TM_QUERY = {
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford FC",
    "Brighton": "Brighton and Hove Albion",
    "Burnley": "Burnley FC",
    "Cardiff": "Cardiff City",
    "Chelsea": "Chelsea FC",
    "Coventry": "Coventry City",
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


def fetch_squad(team_id: int, season_start: int) -> list[dict] | None:
    """One dict per valued player: {player, age, value_eur}. Column order on
    TM's squad table (#, Player, Age, Nat., Current club, Market value) is
    stable across seasons (checked 2015/2020/2025) -- age is always the 2nd
    'zentriert'-classed cell (1st is the squad-number column, which also
    carries the zentriert class)."""
    url = f"https://www.transfermarkt.com/club/kader/verein/{team_id}/plus/0?saison_id={season_start}"
    html = _get(url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="items")
    if table is None or table.find("tbody") is None:
        return None
    players = []
    for row in table.find("tbody").find_all("tr", recursive=False):
        value_cell = row.find("td", class_=lambda c: c and "rechts" in c.split() and "hauptlink" in c.split())
        if value_cell is None:
            continue
        value = _parse_value(value_cell.get_text(strip=True))
        if value is None:
            continue
        posrela = row.find("td", class_="posrela")
        hauptlink_td = posrela.find("td", class_="hauptlink") if posrela else None
        name = hauptlink_td.get_text(strip=True) if hauptlink_td else None
        zentriert_cells = row.find_all("td", class_="zentriert")
        age_text = zentriert_cells[1].get_text(strip=True) if len(zentriert_cells) > 1 else ""
        age = int(age_text) if age_text.isdigit() else None
        players.append({"player": name, "age": age, "value_eur": value})
    return players


PRE_FOTMOB_FIRST_SEASON = 2015  # fotmob.db only covers 2020-21 onward

# 2026-27 has no historical PL results yet (fotmob.db and raw_results both lack
# it), so its 20-team roster has to be given explicitly -- same list used to
# build manual_priors_2026_27.py (RETURNING + PROMOTED), in this script's naming.
CURRENT_SEASON = "2026-2027"
CURRENT_SEASON_ROSTER = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Leeds", "Liverpool", "Man City",
    "Man United", "Newcastle", "Nott'm Forest", "Sunderland", "Tottenham",
    "Hull", "Coventry", "Ipswich",
]


def build_team_season_list(priors_conn: sqlite3.Connection) -> pd.DataFrame:
    """Full PL team-season history: fotmob.db for 2020-21 onward, raw_results
    (priors.db, football-data.co.uk) for 2015-16..2019-20 since fotmob.db
    doesn't go back that far, plus the current season's roster given explicitly
    since it has no historical results yet either way."""
    fotmob_query = """
        SELECT DISTINCT m.season, tm.team_name AS team
        FROM np_matches m JOIN team_id_mapping tm ON tm.team_id = m.home_team
        WHERE m.league_id = 'Premier_League'
        UNION
        SELECT DISTINCT m.season, tm.team_name AS team
        FROM np_matches m JOIN team_id_mapping tm ON tm.team_id = m.away_team
        WHERE m.league_id = 'Premier_League'
    """
    with sqlite3.connect(FOTMOB_DB_PATH) as fconn:
        fotmob_seasons = pd.read_sql(fotmob_query, fconn)
    fotmob_seasons["team"] = fotmob_seasons["team"].replace(FOTMOB_TO_TM_NAME)

    fotmob_min_season = fotmob_seasons["season"].min()
    raw = pd.read_sql("SELECT * FROM raw_results", priors_conn)
    pre_fotmob = raw[
        (raw["division"] == "Premier_League")
        & (raw["season"] >= f"{PRE_FOTMOB_FIRST_SEASON}-{PRE_FOTMOB_FIRST_SEASON + 1}")
        & (raw["season"] < fotmob_min_season)
    ]
    pre_fotmob = pd.concat([
        pre_fotmob[["season", "home_team"]].rename(columns={"home_team": "team"}),
        pre_fotmob[["season", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).drop_duplicates()

    current = pd.DataFrame({"season": CURRENT_SEASON, "team": CURRENT_SEASON_ROSTER})

    team_seasons = pd.concat([pre_fotmob, fotmob_seasons, current], ignore_index=True).drop_duplicates()
    return team_seasons.sort_values(["team", "season"])


def _existing_players(conn: sqlite3.Connection) -> pd.DataFrame:
    cols = ["team", "season", "player", "age", "value_eur"]
    try:
        return pd.read_sql("SELECT * FROM transfermarkt_player_values", conn)
    except pd.errors.DatabaseError:
        return pd.DataFrame(columns=cols)


def _write(conn: sqlite3.Connection, player_rows: list[pd.DataFrame]) -> pd.DataFrame:
    players = pd.concat(player_rows, ignore_index=True)
    players.to_sql("transfermarkt_player_values", conn, if_exists="replace", index=False)

    squads = (
        players.groupby(["team", "season"])["value_eur"]
        .agg(squad_value_eur="sum", n_players="count")
        .reset_index()
    )
    squads.to_sql("transfermarkt_squad_values", conn, if_exists="replace", index=False)
    return players


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    team_seasons = build_team_season_list(conn)
    teams = sorted(team_seasons["team"].unique())
    print(f"{len(teams)} teams, {len(team_seasons)} team-seasons to fetch\n")

    print("Resolving team IDs...")
    team_ids = load_team_ids(teams)
    missing = [t for t in teams if team_ids.get(t) is None]
    if missing:
        print(f"WARNING: could not resolve IDs for: {missing}")

    existing = _existing_players(conn)
    done = set(zip(existing["team"], existing["season"]))

    player_rows = [existing] if len(existing) else []
    print("\nFetching squads (player-level: name, age, value)...")
    for _, r in team_seasons.iterrows():
        team, season = r["team"], r["season"]
        if (team, season) in done:
            continue
        team_id = team_ids.get(team)
        if team_id is None:
            continue
        season_start = int(season.split("-")[0])
        players = fetch_squad(team_id, season_start)
        if not players:
            print(f"  [{team} {season}] FAILED")
            continue
        total = sum(p["value_eur"] for p in players)
        print(f"  [{team} {season}] EUR {total/1e6:.1f}m ({len(players)} valued players)")
        df = pd.DataFrame(players)
        df["team"], df["season"] = team, season
        player_rows.append(df[["team", "season", "player", "age", "value_eur"]])
        _write(conn, player_rows)
        time.sleep(SLEEP)

    final = _write(conn, player_rows) if player_rows else existing
    conn.close()
    n_team_seasons = final[["team", "season"]].drop_duplicates().shape[0]
    print(f"\nSaved {len(final)} player rows across {n_team_seasons} team-seasons "
          f"-> {DB_PATH}::transfermarkt_player_values (+ transfermarkt_squad_values)")
