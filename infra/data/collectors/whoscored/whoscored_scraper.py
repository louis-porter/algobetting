import pandas as pd
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
import main
import seaborn as sns
from datetime import datetime
import sqlite3
import time
import json


def create_driver_with_options():
    options = Options()
    options.set_preference("dom.webnotifications.enabled", False)
    options.set_preference("dom.push.enabled", False)
    driver = webdriver.Firefox(options=options)
    return driver


def dismiss_overlays(driver, timeout=5):
    try:
        close_selectors = [
            ".webpush-swal2-close",
            ".swal2-close",
            "button.webpush-swal2-close",
            "[aria-label='Close']"
        ]
        
        for selector in close_selectors:
            try:
                close_button = WebDriverWait(driver, timeout).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                close_button.click()
                print("✅ Dismissed notification popup")
                time.sleep(0.5)
                return True
            except TimeoutException:
                continue
            except Exception:
                continue
        
    except Exception:
        pass
    
    return False


def safe_click(driver, element, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            element.click()
            return True
        except ElementClickInterceptedException:
            print(f"⚠️  Click blocked by overlay (attempt {attempt + 1}/{max_attempts})")
            dismiss_overlays(driver, timeout=2)
            time.sleep(1)
            if attempt == max_attempts - 1:
                print("💡 Trying JavaScript click as fallback...")
                try:
                    driver.execute_script("arguments[0].click();", element)
                    print("✅ JavaScript click succeeded")
                    return True
                except Exception as e:
                    print(f"❌ JavaScript click failed: {e}")
                    return False
        except Exception as e:
            print(f"❌ Unexpected error during click: {e}")
            if attempt == max_attempts - 1:
                return False
            time.sleep(1)
    
    return False


# ─────────────────────────────────────────────
# MATCH EVENTS
# ─────────────────────────────────────────────

# Columns from the events DataFrame that are scalar / simple types and safe to
# store directly in SQLite.  List/dict columns (qualifiers, satisfiedEventsTypes,
# boolean event-type flags) are handled separately below.
MATCH_EVENTS_SCALAR_COLS = [
    'matchId',
    'startDate',
    'startTime',
    'homeTeam',
    'awayTeam',
    'score',
    'ftScore',
    'htScore',
    'etScore',
    'venueName',
    'maxMinute',
    'eventId',
    'minute',
    'second',
    'teamId',
    'h_a',
    'playerId',
    'playerName',
    'period',
    'type',
    'outcomeType',
    'cardType',
    'isShot',
    'isGoal',
    'shotBodyType',
    'situation',
    'x',
    'y',
    'endX',
    'endY',
]


def _serialize_cell(value):
    """Convert list/dict cells to a JSON string so they can be stored in SQLite."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def prepare_match_events_df(events_df: pd.DataFrame, season_label: str, division: str) -> pd.DataFrame:
    """
    Flatten the raw events DataFrame into a DB-friendly format.

    - Scalar columns listed in MATCH_EVENTS_SCALAR_COLS are kept as-is (missing
      ones are added as NaN so the schema is consistent across matches).
    - 'qualifiers' and 'satisfiedEventsTypes' are JSON-serialised into text columns.
    - Boolean event-type flag columns (e.g. 'Pass', 'BallTouch' ...) are kept as
      integers (0/1) for easy SQL querying.
    - season and division labels are appended.
    - Built with pd.concat to avoid DataFrame fragmentation warnings.

    Returns a copy of the DataFrame ready for to_sql().
    """
    df = events_df.copy()

    # ── Scalar columns ───────────────────────────────────────────────────────
    scalar_data = {}
    for col in MATCH_EVENTS_SCALAR_COLS:
        scalar_data[col] = df[col] if col in df.columns else None

    # ── JSON-serialised list/dict columns ────────────────────────────────────
    for col in ('qualifiers', 'satisfiedEventsTypes'):
        scalar_data[col] = df[col].apply(_serialize_cell) if col in df.columns else None

    # ── Boolean event-type flag columns → int (0/1) ──────────────────────────
    scalar_and_special = set(MATCH_EVENTS_SCALAR_COLS) | {'qualifiers', 'satisfiedEventsTypes'}
    flag_cols = [c for c in df.columns if c not in scalar_and_special]
    flag_data = {}
    for col in flag_cols:
        try:
            flag_data[col] = df[col].astype('boolean').fillna(False).astype(int)
        except Exception:
            flag_data[col] = df[col]

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta_data = {
        'season': season_label,
        'division': division,
    }

    # Build in one concat to avoid fragmentation
    result = pd.concat(
        [pd.DataFrame(scalar_data, index=df.index),
         pd.DataFrame(flag_data, index=df.index),
         pd.DataFrame(meta_data, index=df.index)],
        axis=1
    )

    # Ensure isShot/isGoal are plain int
    for col in ('isShot', 'isGoal'):
        if col in result.columns:
            result[col] = result[col].astype(int)

    return result


def save_match_events_to_database(
    events_df: pd.DataFrame,
    season_label: str,
    division: str,
    db_name: str = r"/Users/admin/dev/algobetting/infra/data/db/fotmob.db"
):
    """
    Save per-event data to the 'match_events' table in SQLite.

    Deduplication key: (matchId, eventId) — if a (matchId, eventId) pair already
    exists in the table it will be skipped.

    Args:
        events_df   : Raw events DataFrame returned by main.createEventsDF()
                      (before or after addEpvToDataFrame – either works).
        season_label: e.g. '2025-2026'
        division    : e.g. 'Premier_League'
        db_name     : Path to the SQLite database file.
    """
    if events_df.empty:
        print("⚠️  No match events to save (empty DataFrame)")
        return

    table_name = 'match_events'
    df_to_save = prepare_match_events_df(events_df, season_label, division)

    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Add any new columns that exist in df_to_save but not yet in the table
            existing_cols = {row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")}
            for col in df_to_save.columns:
                if col not in existing_cols:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN \"{col}\" INTEGER DEFAULT 0")
                    print(f"➕ Added new column '{col}' to '{table_name}'")
            conn.commit()

            existing_df = pd.read_sql(
                f"SELECT matchId, eventId FROM {table_name}", conn
            )
            existing_keys = set(
                zip(existing_df['matchId'].astype(str), existing_df['eventId'].astype(str))
            )
            new_mask = ~df_to_save.apply(
                lambda row: (str(row['matchId']), str(row['eventId'])) in existing_keys,
                axis=1
            )
            new_df = df_to_save[new_mask]

            duplicates = len(df_to_save) - len(new_df)
            if duplicates:
                print(f"⏭️  Skipped {duplicates} duplicate rows in '{table_name}'")

            if len(new_df) > 0:
                new_df.to_sql(table_name, conn, if_exists='append', index=False)
                print(f"✅ Inserted {len(new_df)} new rows into '{table_name}'")
            else:
                print(f"ℹ️  No new rows to insert into '{table_name}'")
        else:
            df_to_save.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"✅ Created '{table_name}' and inserted {len(df_to_save)} rows")

        print(f"✓ Match events saved to {db_name}")

    except Exception as e:
        import traceback
        print(f"❌ Error saving match events to database: {e}")
        print(traceback.format_exc())
    finally:
        conn.close()


# ─────────────────────────────────────────────
# EPV (existing)
# ─────────────────────────────────────────────

def save_epv_to_database(epv_df, db_name=r"/Users/admin/dev/algobetting/infra/data/db/fotmob.db"):
    if epv_df.empty:
        print("⚠️  No data to save to database (empty DataFrame)")
        return
    
    conn = sqlite3.connect(db_name)
    try:
        table_name = 'epv'
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            existing_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            check_columns = ['matchId', 'team']
            existing_keys = set(existing_df[check_columns].apply(lambda row: tuple(row), axis=1))
            new_rows_mask = ~epv_df[check_columns].apply(lambda row: tuple(row) in existing_keys, axis=1)
            new_df = epv_df[new_rows_mask]
            
            duplicates = len(epv_df) - len(new_df)
            if duplicates > 0:
                print(f"⏭️  Skipped {duplicates} duplicate rows in '{table_name}'")
            
            if len(new_df) > 0:
                new_df.to_sql(table_name, conn, if_exists='append', index=False)
                print(f"✅ Inserted {len(new_df)} new rows to '{table_name}' table")
            else:
                print(f"ℹ️  No new rows to insert in '{table_name}' table")
        else:
            epv_df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"✅ Created '{table_name}' table and inserted {len(epv_df)} rows")
        
        print(f"\n✓ EPV data successfully saved to {db_name}")
        
    except Exception as e:
        print(f"❌ Error saving to database: {str(e)}")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def process_epv_data(start_date, end_date, season='2025/2026',
                     season_label='2025-2026', division='Premier_League'):
    """
    Process EPV data for matches within a date range.
    Also saves a full per-event log to the 'match_events' table.

    Args:
        start_date   : datetime – start of date window
        end_date     : datetime – end of date window
        season       : season string e.g. '2025/2026'
        season_label : label stored in DB e.g. '2025-2026'
        division     : division/league name e.g. 'Premier_League'

    Returns:
        pd.DataFrame: EPV data aggregated by match and team
    """
    print(f"🔍 Processing EPV data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Season: {season}")
    print("-" * 60)
    
    try:
        match_urls = main.getMatchUrls(season=season, start_date=start_date, end_date=end_date)
    except ElementClickInterceptedException:
        print("\n❌ ERROR: Click intercepted by overlay popup")
        return pd.DataFrame()
    except Exception as e:
        print(f"\n❌ ERROR getting match URLs: {str(e)}")
        return pd.DataFrame()
    
    if not match_urls:
        print("\n❌ ERROR: No match URLs found.")
        return pd.DataFrame()
    
    for match in match_urls:
        try:
            match['date_dt'] = datetime.strptime(match['date'], '%A, %b %d %Y')
        except Exception as e:
            print(f"⚠️  Warning: Could not parse date '{match.get('date')}': {e}")
            continue
    
    match_urls = [match for match in match_urls if 'date_dt' in match and start_date <= match['date_dt'] <= end_date]
    
    print(f"\n📅 Found {len(match_urls)} matches in date range")
    
    if len(match_urls) == 0:
        print("\n⚠️  No matches found in the specified date range.")
        print(f"   Start date: {start_date.strftime('%Y-%m-%d')}")
        print(f"   End date: {end_date.strftime('%Y-%m-%d')}")
        return pd.DataFrame()
    
    print("\n📥 Fetching match data...")
    matches_data = main.getMatchesData(match_urls=match_urls)
    
    if not matches_data:
        print("\n⚠️  No match data retrieved")
        return pd.DataFrame()
    
    print(f"✅ Retrieved data for {len(matches_data)} matches")
    
    # ── Build events DataFrames ──────────────────────────────────────────────
    print("\n⚙️  Processing events...")
    events_ls = [main.createEventsDF(match) for match in matches_data]
    
    if not events_ls:
        print("\n⚠️  No events data to process")
        return pd.DataFrame()

    # ── Save raw match events BEFORE EPV calculation (EPV only covers passes) ─
    print("\n📝 Saving match events log to database...")
    all_events_raw = pd.concat(events_ls, ignore_index=True)
    save_match_events_to_database(all_events_raw, season_label=season_label, division=division)

    # ── Add EPV values ───────────────────────────────────────────────────────
    print("\n📊 Calculating EPV values...")
    events_list = [main.addEpvToDataFrame(match) for match in events_ls]
    
    if not events_list:
        print("\n⚠️  No events data after EPV calculation")
        return pd.DataFrame()
    
    events_dfs = pd.concat(events_list)
    
    # ── Aggregate EPV per match/team ─────────────────────────────────────────
    print("🔄 Aggregating EPV data...")
    events_dfs['team'] = events_dfs['homeTeam'].where(events_dfs['h_a'] == 'h', events_dfs['awayTeam'])
    events_dfs['opponent'] = events_dfs['awayTeam'].where(events_dfs['h_a'] == 'h', events_dfs['homeTeam'])
    events_dfs['startDate'] = pd.to_datetime(events_dfs['startDate'])
    
    epv = events_dfs.groupby(["matchId", "team", 'opponent', 'startDate']).agg({
        "EPV": "sum"
    }).reset_index()
    
    epv['season'] = season_label
    epv['division'] = division
    
    team_mapping = {
        'Nottingham Forest': 'Nottm Forest',
        'Manchester City': 'Man City',
        'Manchester United': 'Man United',
    }
    epv['team'] = epv['team'].replace(team_mapping)
    
    print(f"\n✅ Processed {len(epv)} EPV records")
    
    print("\n💾 Saving EPV data to database...")
    save_epv_to_database(epv)
    
    return epv


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    start_date = datetime(2026, 3, 8)
    end_date = datetime(2026, 3, 17)
    
    print("=" * 60)
    print("🏆 WhoScored EPV Data Scraper")
    print("=" * 60)
    
    epv_df = process_epv_data(
        start_date=start_date,
        end_date=end_date,
        season='2025/2026',
        season_label='2025-2026',
        division='Premier_League'
    )
    
    if epv_df.empty:
        print("\n⚠️  FINAL RESULT: No data was collected")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Verify the date range has matches")
        print("   2. Check if the season is correct (2025/2026)")
        print("   3. Try running in non-headless mode to see browser behavior")
        print("   4. Check your internet connection")
    else:
        print(f"\n✅ SUCCESS: Collected and saved {len(epv_df)} EPV records")
        print("\n📋 Sample of collected data:")
        print(epv_df.head())