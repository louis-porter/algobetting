"""
Orchestrates the full data collection pipeline:
  1. FotMob season downloader  — fetches raw match JSON files (skips already-downloaded)
  2. FotMob ETL (standard)     — parses JSON → SQLite (matches, shots, etc.)
  2b. FotMob ETL (non-penalty) — parses JSON → SQLite (np_shots, np_matches, red_cards, etc.)
  3. WhoScored EPV             — Selenium scrape → EPV + shot possession IDs in SQLite

Each step runs independently — a failure in one is logged and the pipeline moves on to
the next rather than dying silently, since this is meant to run unattended (cron/launchd).
Every run appends to _run_log.txt next to this file, and on macOS a failure fires a
native notification (nothing fires on success — the log is enough for a healthy run).

Usage:
    python _collect_all_strength.py [--season 2025-2026] [--start 2026-03-25] [--end 2026-03-26]

Defaults:
    --season  : current calendar-year pair based on today's date
    --start   : 7 days ago
    --end     : today
"""

import argparse
import subprocess
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# whoscored_scraper.py does a bare `import main` expecting whoscored/ to be on the path
_collectors = Path(__file__).parent
sys.path.insert(0, str(_collectors))
sys.path.insert(0, str(_collectors / "whoscored"))

from fotmob.fotmob_season_downloader import store_season, LEAGUES
from fotmob.fotmob_etl_database import main as fotmob_etl_main
from fotmob.fotmob_etl_database_non_penalty import main as fotmob_etl_np_main
from fotmob.assign_gameweeks import write_to_db as assign_gameweeks_to_db
from whoscored.whoscored_scraper import process_epv_data
from whoscored.add_epv_to_events import add_epv_to_events_table

RUN_LOG_PATH = _collectors / "_run_log.txt"


class _Tee:
    """Duplicates writes to the real stream and a log file, so existing print()
    calls throughout the pipeline show up both on the console and in the run log
    without having to rewrite them as logging calls."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()


def _notify_macos(title: str, message: str):
    """Best-effort native notification. No-op (and silent) off macOS or if it fails —
    this is a courtesy, not something the pipeline's success should depend on."""
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
            check=False,
            timeout=10,
        )
    except Exception:
        pass


def run_step(name: str, failures: list, fn, *args, **kwargs):
    """Run one pipeline step, catching and logging any exception instead of letting
    it kill the rest of the pipeline — later steps are largely independent of earlier
    ones (WhoScored doesn't depend on FotMob at all), so one bad step shouldn't mean
    zero data collected on an unattended run."""
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"\n❌ Step failed: {name}\n{traceback.format_exc()}")
        failures.append((name, str(e)))


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
    failures: list[tuple[str, str]] = []

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
        run_step("FotMob JSON download", failures, store_season, league_cfg, season_start)
    else:
        print("── Step 1: FotMob JSON download [SKIPPED] ────────────────────")

    # ── Step 2: FotMob ETL ────────────────────────────────────────────────────
    if not args.skip_fotmob_etl:
        print("\n── Step 2a: FotMob ETL (standard) → SQLite ───────────────────")
        run_step("FotMob ETL (standard)", failures, fotmob_etl_main, season=season_label, league=args.league)
        print("\n── Step 2b: FotMob ETL (non-penalty) → SQLite ────────────────")
        run_step("FotMob ETL (non-penalty)", failures, fotmob_etl_np_main, season=season_label, league=args.league)
        print("\n── Step 2c: Assign custom gameweeks ──────────────────────────")
        run_step("Assign gameweeks", failures, assign_gameweeks_to_db, league=args.league, season=season_label)
    else:
        print("\n── Step 2: FotMob ETL [SKIPPED] ──────────────────────────────")

    # ── Step 3: WhoScored EPV ─────────────────────────────────────────────────
    if not args.skip_whoscored:
        print("\n── Step 3: WhoScored EPV scrape ──────────────────────────────")
        run_step(
            "WhoScored EPV scrape", failures, process_epv_data,
            start_date=start_date,
            end_date=end_date,
            season=whoscored_season,
            season_label=season_label,
            division=args.league,
        )
        print("\n── Step 3b: Write EPV values to match_events ─────────────────")
        run_step("Write EPV to match_events", failures, add_epv_to_events_table)
    else:
        print("\n── Step 3: WhoScored EPV [SKIPPED] ───────────────────────────")

    # ── Summary ────────────────────────────────────────────────────────────────
    if failures:
        print(f"\n⚠️  Completed with {len(failures)} failed step(s):")
        for name, err in failures:
            print(f"   - {name}: {err}")
        _notify_macos(
            "AlgoBetting pipeline: failures",
            f"{len(failures)} step(s) failed — see _run_log.txt: " + ", ".join(n for n, _ in failures),
        )
    else:
        print("\n✅ Done.\n")

    return len(failures)


if __name__ == "__main__":
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as _log_file:
        _log_file.write(f"\n{'='*60}\n{datetime.now().isoformat()} — starting run: {' '.join(sys.argv[1:])}\n{'='*60}\n")
        sys.stdout = _Tee(sys.stdout, _log_file)
        sys.stderr = _Tee(sys.stderr, _log_file)
        try:
            n_failures = main()
        finally:
            sys.stdout, sys.stderr = sys.stdout._stream, sys.stderr._stream

    sys.exit(1 if n_failures else 0)
 
 