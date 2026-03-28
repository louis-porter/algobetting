"""
Orchestrates the full data collection pipeline:
  1. FotMob season downloader  — fetches raw match JSON files (skips already-downloaded)
  2. FotMob ETL                — parses JSON → SQLite (np_shots, np_matches, red_cards, etc.)
  3. WhoScored EPV             — Selenium scrape → EPV + shot possession IDs in SQLite

Usage:
    python _collect_all_strength.py [--season 2025-2026] [--start 2026-03-25] [--end 2026-03-26] [--xmas <header>]

Defaults:
    --season  : current calendar-year pair based on today's date
    --start   : 7 days ago
    --end     : today
    --xmas    : prompted interactively if not supplied
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# whoscored_scraper.py does a bare `import main` expecting whoscored/ to be on the path
_collectors = Path(__file__).parent
sys.path.insert(0, str(_collectors))
sys.path.insert(0, str(_collectors / "whoscored"))

from fotmob.fotmob_season_downloader import store_season, LEAGUES
from fotmob.fotmob_etl_database_non_penalty import main as fotmob_etl_main
from whoscored.whoscored_scraper import process_epv_data


def _default_season():
    """Returns e.g. '2025-2026' based on today. Seasons start in August."""
    today = datetime.today()
    year = today.year
    if today.month >= 8:
        return f"{year}-{year + 1}"
    else:
        return f"{year - 1}-{year}"


def _parse_args():
    parser = argparse.ArgumentParser(description="Collect FotMob + WhoScored data")

    parser.add_argument(
        "--league",
        default="Premier_League",
        help="League name (must match an entry in fotmob_season_downloader.LEAGUES). Default: Premier_League",
    )
    parser.add_argument(
        "--season",
        default=_default_season(),
        help="Season in YYYY-YYYY format, e.g. 2025-2026. Default: current season.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="WhoScored start date YYYY-MM-DD. Default: 7 days ago.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="WhoScored end date YYYY-MM-DD. Default: today.",
    )
    parser.add_argument(
        "--skip-fotmob-download",
        action="store_true",
        help="Skip the FotMob JSON download step (ETL still runs).",
    )
    parser.add_argument(
        "--skip-fotmob-etl",
        action="store_true",
        help="Skip the FotMob ETL step.",
    )
    parser.add_argument(
        "--skip-whoscored",
        action="store_true",
        help="Skip the WhoScored EPV scraping step.",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    # ── Derive consistent formats from --season ──────────────────────────────
    # e.g. '2025-2026' → season_start=2025, whoscored_season='2025/2026'
    try:
        start_year_str, end_year_str = args.season.split("-")
        season_start = int(start_year_str)
    except ValueError:
        raise ValueError(f"--season must be in YYYY-YYYY format, got: {args.season!r}")

    whoscored_season = f"{start_year_str}/{end_year_str}"   # '2025/2026'
    season_label = args.season                               # '2025-2026'

    # ── Date window ───────────────────────────────────────────────────────────
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        # Default to the beginning of the season (Aug 1 for winter leagues)
        start_date = datetime(season_start, 7, 1)

    # ── Resolve league config ─────────────────────────────────────────────────
    league_configs = {l["name"]: l for l in LEAGUES}
    if args.league not in league_configs:
        available = list(league_configs.keys())
        raise ValueError(
            f"League {args.league!r} not found in LEAGUES. "
            f"Available (uncomment in fotmob_season_downloader.py): {available}"
        )
    league_cfg = league_configs[args.league]

    print(f"\n{'='*60}")
    print(f"  League  : {args.league}")
    print(f"  Season  : {args.season}")
    print(f"  Window  : {start_date.date()} → {end_date.date()}")
    print(f"{'='*60}\n")

    # ── Step 1: FotMob download ───────────────────────────────────────────────
    if not args.skip_fotmob_download:
        print("── Step 1: FotMob JSON download ──────────────────────────────")
        store_season(league_cfg, season_start)
    else:
        print("── Step 1: FotMob JSON download [SKIPPED] ────────────────────")

    # ── Step 2: FotMob ETL ────────────────────────────────────────────────────
    if not args.skip_fotmob_etl:
        print("\n── Step 2: FotMob ETL → SQLite ───────────────────────────────")
        fotmob_etl_main(season=season_label, league=args.league)
    else:
        print("\n── Step 2: FotMob ETL [SKIPPED] ──────────────────────────────")

    # ── Step 3: WhoScored EPV ─────────────────────────────────────────────────
    if not args.skip_whoscored:
        print("\n── Step 3: WhoScored EPV scrape ──────────────────────────────")
        process_epv_data(
            start_date=start_date,
            end_date=end_date,
            season=whoscored_season,
            season_label=season_label,
            division=args.league,
        )
    else:
        print("\n── Step 3: WhoScored EPV [SKIPPED] ───────────────────────────")

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
 