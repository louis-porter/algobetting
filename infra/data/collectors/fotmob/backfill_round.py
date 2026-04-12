"""
Adds `round` column to `matches` and `np_matches` tables and backfills it
from the on-disk JSON files.

Run once from the repo root:
    python infra/data/collectors/fotmob/backfill_round.py
"""

import json
import sqlite3
from pathlib import Path

DB_PATH  = Path("infra/data/db/fotmob.db")
JSON_DIR = Path("infra/data/json")


def extract_round(data: dict) -> str | None:
    """Return the round/GW number from either JSON format."""
    # New format: general.matchRound
    if "general" in data:
        val = data["general"].get("matchRound") or data["general"].get("leagueRoundName")
        if val is not None:
            return str(val)
    # Old format: matchFacts.infoBox.Tournament.round
    try:
        val = data["matchFacts"]["infoBox"]["Tournament"].get("round")
        if val is not None:
            return str(val)
    except (KeyError, TypeError):
        pass
    return None


def build_round_map(json_dir: Path) -> dict[int, str]:
    """Walk all JSON files and return {match_id: round_str}."""
    round_map: dict[int, str] = {}
    json_files = list(json_dir.rglob("*.json"))
    print(f"Scanning {len(json_files)} JSON files...")

    for fp in json_files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)

            # Normalise nested format
            if "matchFacts" not in data and "content" in data:
                data = {**data, **data["content"]}

            # Get match_id from either format
            match_id = None
            if "general" in data:
                match_id = data["general"].get("matchId")
            if match_id is None and "matchFacts" in data:
                match_id = data["matchFacts"].get("matchId")

            if match_id is None:
                continue

            round_val = extract_round(data)
            if round_val is not None:
                round_map[int(match_id)] = round_val

        except Exception as e:
            print(f"  Skipped {fp.name}: {e}")

    print(f"Found round data for {len(round_map)} matches.")
    return round_map


def migrate_table(conn: sqlite3.Connection, table: str, round_map: dict[int, str]) -> None:
    cur = conn.cursor()

    # Add column if missing
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if "round" not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN round TEXT")
        print(f"  Added `round` column to `{table}`.")
    else:
        print(f"  `round` column already exists in `{table}`, skipping ALTER.")

    # Fetch rows that need updating
    cur.execute(f"SELECT match_id FROM {table} WHERE round IS NULL")
    null_ids = [row[0] for row in cur.fetchall()]
    print(f"  {len(null_ids)} rows in `{table}` need backfill.")

    updated = 0
    for match_id in null_ids:
        round_val = round_map.get(match_id)
        if round_val is not None:
            cur.execute(f"UPDATE {table} SET round = ? WHERE match_id = ?", (round_val, match_id))
            updated += 1

    conn.commit()
    missing = len(null_ids) - updated
    print(f"  Updated {updated} rows. {missing} rows had no matching JSON.")


def main() -> None:
    round_map = build_round_map(JSON_DIR)

    conn = sqlite3.connect(DB_PATH)
    try:
        for table in ("matches", "np_matches"):
            print(f"\nMigrating `{table}`...")
            migrate_table(conn, table, round_map)
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
