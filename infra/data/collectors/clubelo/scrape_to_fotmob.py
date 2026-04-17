"""
Scrape ClubElo ratings weekly (every Tuesday) from 2017 to today
and store them in fotmob.db as the `clubelo_ratings` table.

Safe to re-run — skips dates already present.
"""

import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = '/Users/admin/dev/algobetting/infra/data/db/fotmob.db'
TABLE_NAME = 'clubelo_ratings'
BASE_URL = 'http://api.clubelo.com/'
START_DATE = '2017-06-01'
DELAY_SECONDS = 1
MAX_RETRIES = 3


def get_tuesdays(start_date: datetime, end_date: datetime) -> list[datetime]:
    current = start_date
    while current.weekday() != 1:  # Tuesday = 1
        current += timedelta(days=1)
    tuesdays = []
    while current <= end_date:
        tuesdays.append(current)
        current += timedelta(days=7)
    return tuesdays


def fetch_date(session: requests.Session, date_str: str) -> str | None:
    url = BASE_URL + date_str
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=(10, 30), headers={'Accept': 'text/csv, */*'})
            r.raise_for_status()
            if r.text.strip():
                return r.text
            logger.warning(f"{date_str}: empty response")
            return None
        except requests.RequestException as e:
            logger.warning(f"{date_str}: attempt {attempt + 1} failed — {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(DELAY_SECONDS * 2)
    return None


def existing_dates(conn: sqlite3.Connection) -> set[str]:
    try:
        rows = conn.execute(f"SELECT DISTINCT fetch_date FROM {TABLE_NAME}").fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()


def main():
    start = datetime.strptime(START_DATE, '%Y-%m-%d')
    end = datetime.now()
    tuesdays = get_tuesdays(start, end)
    logger.info(f"Total Tuesdays to process: {len(tuesdays)}")

    conn = sqlite3.connect(DB_PATH)
    seen = existing_dates(conn)
    logger.info(f"Already in DB: {len(seen)} dates")

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    processed = skipped = failed = 0

    for i, tuesday in enumerate(tuesdays, 1):
        date_str = tuesday.strftime('%Y-%m-%d')

        if date_str in seen:
            skipped += 1
            continue

        logger.info(f"[{i}/{len(tuesdays)}] Fetching {date_str}")
        csv_text = fetch_date(session, date_str)

        if csv_text is None:
            logger.error(f"Failed: {date_str}")
            failed += 1
            continue

        try:
            df = pd.read_csv(StringIO(csv_text))
            df['fetch_date'] = date_str
            df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
            conn.commit()
            seen.add(date_str)
            processed += 1
        except Exception as e:
            logger.error(f"DB error for {date_str}: {e}")
            failed += 1

        time.sleep(DELAY_SECONDS)

    conn.close()
    logger.info(f"Done. Processed: {processed}, Skipped: {skipped}, Failed: {failed}")

    # Quick summary
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    dates = conn.execute(f"SELECT COUNT(DISTINCT fetch_date) FROM {TABLE_NAME}").fetchone()[0]
    conn.close()
    logger.info(f"Table '{TABLE_NAME}': {total:,} rows across {dates} dates")


if __name__ == '__main__':
    main()
