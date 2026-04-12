"""
Assigns custom gameweek numbers to matches based on calendar window rules:

  Weekend window : Friday → Monday   (any Fri/Sat/Sun/Mon)
  Midweek window : Tuesday → Thursday (any Tue/Wed/Thu)

Every distinct window gets its own GW number. Rescheduled or postponed
fixtures that fall in a small midweek window are kept separate — merging
them into an adjacent GW would mean a team could appear twice in one GW,
causing predictions to be built on data that wasn't yet available at the
time of the earlier matches.

Run standalone from the repo root to write custom_gw to the DB:
    python infra/data/collectors/fotmob/assign_gameweeks.py

Or import assign_gameweeks(df) for use in notebooks.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pandas as pd

DB_PATH = "infra/data/db/fotmob.db"

# Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
_WEEKEND_DAYS = {0, 4, 5, 6}   # Mon, Fri, Sat, Sun
_MIDWEEK_DAYS = {1, 2, 3}      # Tue, Wed, Thu

_DAYS_SINCE_WINDOW_START = {
    # Weekend: anchor to Friday
    4: 0, 5: 1, 6: 2, 0: 3,   # Fri=+0, Sat=+1, Sun=+2, Mon=+3
    # Midweek: anchor to Tuesday
    1: 0, 2: 1, 3: 2,          # Tue=+0, Wed=+1, Thu=+2
}


def _window_start(d: "date") -> tuple["date", str]:
    """Return (window_anchor_date, 'weekend'|'midweek') for a given date."""
    dow = d.weekday()
    wtype = "weekend" if dow in _WEEKEND_DAYS else "midweek"
    return d - timedelta(days=_DAYS_SINCE_WINDOW_START[dow]), wtype


def assign_gameweeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``custom_gw`` column to *df*.

    Each distinct calendar window (Fri-Mon or Tue-Thu) gets its own sequential
    GW number. Rescheduled fixtures in a lone midweek slot are kept as their
    own GW rather than merged, so no team ever appears twice in one GW.

    Parameters
    ----------
    df : DataFrame with at least a ``match_date`` column (str or datetime).

    Returns
    -------
    DataFrame with ``custom_gw`` (int) and ``window_start`` / ``window_type``
    helper columns added.
    """
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])

    # Tag each match with its window anchor
    parsed = df["match_date"].dt.date.apply(lambda d: pd.Series(_window_start(d)))
    df["window_start"] = parsed[0]
    df["window_type"]  = parsed[1]

    # Assign sequential GW numbers to each distinct window in chronological order
    distinct_windows = sorted(df["window_start"].unique())
    gw_map = {ws: i + 1 for i, ws in enumerate(distinct_windows)}

    df["custom_gw"] = df["window_start"].map(gw_map)
    return df


def write_to_db(
    league: str,
    season: str,
    db_path: str = DB_PATH,
) -> None:
    """Compute custom_gw for *league*/*season* and write it back to the DB."""
    conn = sqlite3.connect(db_path)

    for table in ("matches", "np_matches"):
        df = pd.read_sql(
            "SELECT * FROM {} WHERE league_id = ? AND season = ?".format(table),
            conn,
            params=[league, season],
        )

        if df.empty:
            print(f"  No rows in {table} for {league} {season}, skipping.")
            continue

        df = assign_gameweeks(df)

        # Add column if missing
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        existing_cols = {r[1] for r in cur.fetchall()}
        if "custom_gw" not in existing_cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN custom_gw INTEGER")
            conn.commit()
            print(f"  Added custom_gw column to {table}.")

        # Write back row by row
        for _, row in df.iterrows():
            conn.execute(
                f"UPDATE {table} SET custom_gw = ? WHERE match_id = ? AND league_id = ? AND season = ?",
                (int(row["custom_gw"]), int(row["match_id"]), league, season),
            )

        conn.commit()

        # Summary
        summary = (
            df.groupby("custom_gw")
            .agg(first=("match_date", "min"), last=("match_date", "max"), n=("match_id", "count"))
            .sort_index()
        )
        print(f"\n{table} — {league} {season}:")
        print(summary.to_string())

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    write_to_db(league="Premier_League", season="2025-2026")
    write_to_db(league="Superligaen",    season="2025-2026")
