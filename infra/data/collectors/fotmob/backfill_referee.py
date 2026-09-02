"""
Adds a `referee` column to `match_stats` and backfills it from the on-disk
FotMob JSON files (matchFacts.infoBox.Referee.text).

Referee is missing for most of Championship 2020-2021 and about half of
2021-2022 (FotMob didn't report it that far back for that league) — those
rows are left NULL rather than guessed.

Run once from the repo root:
    python infra/data/collectors/fotmob/backfill_referee.py
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("infra/data/db/fotmob.db")
JSON_DIR = Path("infra/data/json")


def extract_referee(data: dict) -> str | None:
    try:
        text = data["matchFacts"]["infoBox"]["Referee"].get("text")
        return text or None
    except (KeyError, TypeError):
        return None


def build_referee_map(json_dir: Path) -> dict[int, str]:
    """Walk all JSON files and return {match_id: referee_name}."""
    referee_map: dict[int, str] = {}
    non_match_dirs = {"api_football", "fbref", "odds"}
    json_files = [
        fp for fp in json_dir.rglob("*.json")
        if fp.relative_to(json_dir).parts[0] not in non_match_dirs
    ]
    print(f"Scanning {len(json_files)} JSON files...")

    for fp in json_files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)

            if "matchFacts" not in data and "content" in data:
                data = {**data, **data["content"]}

            match_id = None
            if "general" in data:
                match_id = data["general"].get("matchId")
            if match_id is None and "matchFacts" in data:
                match_id = data["matchFacts"].get("matchId")
            if match_id is None:
                continue

            referee = extract_referee(data)
            if referee is not None:
                referee_map[int(match_id)] = referee

        except Exception as e:
            print(f"  Skipped {fp.name}: {e}")

    print(f"Found a referee for {len(referee_map)} matches.")
    return referee_map


def migrate_table(conn: sqlite3.Connection, table: str, referee_map: dict[int, str]) -> None:
    cur = conn.cursor()

    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if "referee" not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN referee TEXT")
        print(f"  Added `referee` column to `{table}`.")
    else:
        print(f"  `referee` column already exists in `{table}`, skipping ALTER.")

    cur.execute(f"SELECT match_id FROM {table} WHERE referee IS NULL")
    null_ids = [row[0] for row in cur.fetchall()]
    print(f"  {len(null_ids)} rows in `{table}` need backfill.")

    updated = 0
    for match_id in null_ids:
        referee = referee_map.get(match_id)
        if referee is not None:
            cur.execute(f"UPDATE {table} SET referee = ? WHERE match_id = ?", (referee, match_id))
            updated += 1

    conn.commit()
    missing = len(null_ids) - updated
    print(f"  Updated {updated} rows. {missing} rows had no referee in their JSON.")


def main() -> None:
    referee_map = build_referee_map(JSON_DIR)

    conn = sqlite3.connect(DB_PATH)
    try:
        print("\nMigrating `match_stats`...")
        migrate_table(conn, "match_stats", referee_map)
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
