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

    Deduplication key: (matchId, eventId).
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
# SHOT POSSESSION IDs
# ─────────────────────────────────────────────

SEQUENCE_BREAKS = {
    'CornerAwarded',
    'OffsideGiven',
    'PenaltyFaced',
    'KeeperPickup',
    'Claim',
    'Card',
    'SubstitutionOff',
    'SubstitutionOn',
}
 
SHOT_TYPES = {
    'AttemptSaved', 'Miss', 'Post', 'Goal',
    'MissedShots', 'SavedShot', 'ShotOnPost'
}
 
EVENT_WINDOW = 12
TIME_WINDOW = 20  # seconds

PERIOD_MAP = {
    'FirstHalf': 1,
    'SecondHalf': 2,
    'FirstPeriodOfExtraTime': 3,
    'SecondPeriodOfExtraTime': 4,
    'PenaltyShootout': 5,
}
 
 
def assign_possession_ids(
    match_ids: list,
    db_name: str = r"/Users/admin/dev/algobetting/infra/data/db/fotmob.db"
):
    """
    Read match_events for the given match IDs, assign possession IDs to shots,
    and save results to the shot_possession_id table.
 
    A new possession sequence starts when ANY of the following are true:
      - The period changes (half-time, extra time etc.)
      - A hard-break event occurs (CornerAwarded, OffsideGiven, Card, etc.)
      - More than EVENT_WINDOW events have passed since the same team's last shot
      - More than TIME_WINDOW seconds have elapsed since the same team's last shot
      - The previous shot in this sequence was a Goal (goal is included, next shot is not)
 
    Deduplication key: (matchId, eventId).
 
    Args:
        match_ids : List of matchId values to process
        db_name   : Path to the SQLite database file
    """
    if not match_ids:
        print("⚠️  No match IDs provided for possession ID assignment")
        return
 
    conn = sqlite3.connect(db_name)
    try:
        ids_placeholder = ",".join(f"'{m}'" for m in match_ids)
        events_df = pd.read_sql(
            f"SELECT * FROM match_events WHERE matchId IN ({ids_placeholder})",
            conn
        )
 
        if events_df.empty:
            print("⚠️  No match events found for provided match IDs")
            return
 
        print(f"\n⚙️  Assigning possession IDs for {events_df['matchId'].nunique()} matches...")
 
        events_df = events_df.sort_values(
            ['matchId', 'period', 'minute', 'second']
        ).reset_index(drop=True)

        events_df['minute'] = pd.to_numeric(events_df['minute'], errors='coerce').fillna(0).astype(int)
        events_df['second'] = pd.to_numeric(events_df['second'], errors='coerce').fillna(0).astype(int)
 
        all_shot_possessions = []
        sequence_counter = 0
 
        for match_id, match_events in events_df.groupby('matchId'):
            match_events = match_events.reset_index(drop=True)
 
            current_sequence = None
            last_shot_idx = {}   # teamId -> event index of their last shot
            last_shot_time = {}  # teamId -> encoded time of their last shot
            last_period = None
 
            for idx, row in match_events.iterrows():
                event_type = row.get('type')
                team_id = row.get('teamId')
                period = row.get('period')
 
                # Encode current event time — period offset ensures times from
                # different periods are never compared as close together
                period_int = PERIOD_MAP.get(period, 0)
                current_time = (
                    period_int * 10000
                    + int(row.get('minute') or 0) * 60
                    + int(row.get('second') or 0)
                )
 
                # Period change — hard reset everything
                if period != last_period:
                    sequence_counter += 1
                    current_sequence = sequence_counter
                    last_shot_idx = {}
                    last_shot_time = {}
                    last_period = period
 
                # Hard break events — reset sequence for all teams
                if event_type in SEQUENCE_BREAKS:
                    sequence_counter += 1
                    current_sequence = sequence_counter
                    last_shot_idx = {}
                    last_shot_time = {}
 
                # Only tag shots from here
                if event_type not in SHOT_TYPES:
                    continue
 
                last_idx = last_shot_idx.get(team_id)
                last_time = last_shot_time.get(team_id)
 
                # Time elapsed since this team's last shot (None if no prior shot)
                time_elapsed = (current_time - last_time) if last_time is not None else None
 
                # New sequence if:
                #   - no prior shot from this team in this match/period
                #   - too many events have passed
                #   - more than 20 seconds have elapsed
                if (last_idx is None
                        or (idx - last_idx) > EVENT_WINDOW
                        or time_elapsed is None
                        or time_elapsed > TIME_WINDOW):
                    sequence_counter += 1
                    current_sequence = sequence_counter
 
                # Record this shot's index and time before potentially resetting
                last_shot_idx[team_id] = idx
                last_shot_time[team_id] = current_time
 
                all_shot_possessions.append({
                    'matchId': match_id,
                    'eventId': row.get('eventId'),
                    'teamId': team_id,
                    'period': period,
                    'minute': row.get('minute'),
                    'second': row.get('second'),
                    'playerName': row.get('playerName'),
                    'eventType': event_type,
                    'possession_id': f"{match_id}_{current_sequence}",
                    'season': row.get('season'),
                    'division': row.get('division'),
                })
 
                # Goal is included in this sequence but ends it —
                # the next shot must start a fresh sequence
                if event_type == 'Goal':
                    sequence_counter += 1
                    current_sequence = sequence_counter
                    last_shot_idx = {}
                    last_shot_time = {}
 
        result_df = pd.DataFrame(all_shot_possessions)

        # Merge penalty flags back from events_df
        penalty_cols = ['eventId', 'matchId', 'penaltyScored', 'keeperPenaltySaved', 'penaltyMissed']
        available_penalty_cols = [c for c in penalty_cols if c in events_df.columns]
        result_df = result_df.merge(
            events_df[available_penalty_cols].drop_duplicates(subset=['eventId', 'matchId']),
            on=['eventId', 'matchId'],
            how='left'
        )

        # Filter out penalties
        penalty_flag_cols = [c for c in ['penaltyScored', 'keeperPenaltySaved', 'penaltyMissed'] if c in result_df.columns]
        if penalty_flag_cols:
            penalty_mask = result_df[penalty_flag_cols].eq(1).any(axis=1)
            result_df = result_df[~penalty_mask]
            result_df = result_df.drop(columns=penalty_flag_cols)

        result_df['shot_rank'] = result_df.groupby('matchId').cumcount() + 1
 
        # ── Save with deduplication ───────────────────────────────────────────
        table_name = 'shot_possession_id'
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        table_exists = cursor.fetchone() is not None
 
        if table_exists:
            existing_df = pd.read_sql(f"SELECT matchId, eventId FROM {table_name}", conn)
            existing_keys = set(
                zip(existing_df['matchId'].astype(str), existing_df['eventId'].astype(str))
            )
            new_mask = ~result_df.apply(
                lambda row: (str(row['matchId']), str(row['eventId'])) in existing_keys,
                axis=1
            )
            new_df = result_df[new_mask]
 
            duplicates = len(result_df) - len(new_df)
            if duplicates:
                print(f"⏭️  Skipped {duplicates} duplicate rows in '{table_name}'")
 
            if len(new_df) > 0:
                new_df.to_sql(table_name, conn, if_exists='append', index=False)
                print(f"✅ Inserted {len(new_df)} new rows into '{table_name}'")
            else:
                print(f"ℹ️  No new rows to insert into '{table_name}'")
        else:
            result_df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"✅ Created '{table_name}' and inserted {len(result_df)} rows")
 
    except Exception as e:
        import traceback
        print(f"❌ Error assigning possession IDs: {e}")
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
    Also saves a full per-event log to the 'match_events' table and assigns
    possession IDs to shots via the 'shot_possession_id' table.

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

    # ── Save raw match events BEFORE EPV calculation ─────────────────────────
    print("\n📝 Saving match events log to database...")
    all_events_raw = pd.concat(events_ls, ignore_index=True)
    save_match_events_to_database(all_events_raw, season_label=season_label, division=division)

    # ── Assign possession IDs to shots in the just-scraped matches ────────────
    print("\n🔗 Assigning shot possession IDs...")
    scraped_match_ids = all_events_raw['matchId'].unique().tolist()
    assign_possession_ids(scraped_match_ids)

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
    DB_PATH = r"/Users/admin/dev/algobetting/infra/data/db/fotmob.db"
    conn = sqlite3.connect(DB_PATH)
    match_ids = pd.read_sql("SELECT DISTINCT matchId FROM match_events", conn)['matchId'].tolist()
    conn.close()

    print(f"Found {len(match_ids)} matches to process...")
    assign_possession_ids(match_ids, db_name=DB_PATH)
    print("Done.")

# if __name__ == "__main__":
#     start_date = datetime(2026, 3, 19)
#     end_date = datetime(2026, 3, 20)
    
#     print("=" * 60)
#     print("🏆 WhoScored EPV Data Scraper")
#     print("=" * 60)
    
#     epv_df = process_epv_data(
#         start_date=start_date,
#         end_date=end_date,
#         season='2025/2026',
#         season_label='2025-2026',
#         division='Premier_League'
#     )
    
#     if epv_df.empty:
#         print("\n⚠️  FINAL RESULT: No data was collected")
#         print("\n🔧 Troubleshooting steps:")
#         print("   1. Verify the date range has matches")
#         print("   2. Check if the season is correct (2025/2026)")
#         print("   3. Try running in non-headless mode to see browser behavior")
#         print("   4. Check your internet connection")
#     else:
#         print(f"\n✅ SUCCESS: Collected and saved {len(epv_df)} EPV records")
#         print("\n📋 Sample of collected data:")
#         print(epv_df.head())