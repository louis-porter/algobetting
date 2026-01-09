import pandas as pd
import matplotlib.pyplot as plt
from selenium import webdriver
import main
import seaborn as sns
from datetime import datetime
import sqlite3

def save_epv_to_database(epv_df, db_name=r"/Users/admin/Documents/dev/algobetting/infra/data/db/fotmob.db"):
    """
    Save the EPV DataFrame to SQLite database table with duplicate checking.
    Args:
        epv_df (pd.DataFrame): EPV data
        db_name (str): Name of the SQLite database file
    """
    # Check if dataframe is empty
    if epv_df.empty:
        print("⚠️  No data to save to database (empty DataFrame)")
        return
    
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    try:
        table_name = 'epv'
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Read existing data
            existing_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            # Key columns for duplicate checking - just matchId and team
            check_columns = ['matchId', 'team']
            # Create composite keys for comparison
            existing_keys = set(existing_df[check_columns].apply(lambda row: tuple(row), axis=1))
            new_rows_mask = ~epv_df[check_columns].apply(lambda row: tuple(row) in existing_keys, axis=1)
            new_df = epv_df[new_rows_mask]
            
            rows_before = len(epv_df)
            rows_after = len(new_df)
            duplicates = rows_before - rows_after
            
            if duplicates > 0:
                print(f"⏭️  Skipped {duplicates} duplicate rows in '{table_name}'")
            
            # Insert only new rows
            if len(new_df) > 0:
                new_df.to_sql(table_name, conn, if_exists='append', index=False)
                print(f"✅ Inserted {len(new_df)} new rows to '{table_name}' table")
            else:
                print(f"ℹ️  No new rows to insert in '{table_name}' table")
        else:
            # Table doesn't exist, insert all rows
            epv_df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"✅ Created '{table_name}' table and inserted {len(epv_df)} rows")
        
        print(f"\n✓ EPV data successfully saved to {db_name}")
        
    except Exception as e:
        print(f"❌ Error saving to database: {str(e)}")
    finally:
        conn.close()

def process_epv_data(start_date, end_date, competition='england-premier-league', season='2025/2026', 
                     season_label='2025-2026', division='Premier_League'):
    """
    Process EPV data for matches within a date range.
    
    Args:
        start_date (datetime): Start date for filtering matches
        end_date (datetime): End date for filtering matches
        competition (str): Competition identifier (default: 'england-premier-league')
        season (str): Season in format 'YYYY/YYYY' (default: '2025/2026')
        season_label (str): Season label for database (default: '2025-2026')
        division (str): Division/league name (default: 'Premier_League')
    
    Returns:
        pd.DataFrame: EPV data aggregated by match and team
    """
    print(f"🔍 Processing EPV data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Competition: {competition}, Season: {season}")
    print("-" * 60)
    
    league_urls = main.getLeagueUrls()
    match_urls = main.getMatchUrls(comp_urls=league_urls, competition=competition, season=season)
    
    # Check if match_urls is empty or None
    if not match_urls:
        print("\n❌ ERROR: No match URLs found. This could be due to:")
        print("   1. Season not available for this competition")
        print("   2. Overlay blocking issues preventing data collection")
        print("   3. Incorrect competition or season name")
        return pd.DataFrame()
    
    # Convert dates for ALL matches
    for match in match_urls:
        try:
            match['date_dt'] = datetime.strptime(match['date'], '%A, %b %d %Y')
        except Exception as e:
            print(f"⚠️  Warning: Could not parse date '{match.get('date')}': {e}")
            continue
    
    # Filter matches by date range
    match_urls = [match for match in match_urls if 'date_dt' in match and start_date <= match['date_dt'] <= end_date]
    
    print(f"\n📅 Found {len(match_urls)} matches in date range")
    
    # CRITICAL CHECK: Exit if no matches found
    if len(match_urls) == 0:
        print("\n⚠️  No matches found in the specified date range.")
        print(f"   Start date: {start_date.strftime('%Y-%m-%d')}")
        print(f"   End date: {end_date.strftime('%Y-%m-%d')}")
        print("\n💡 Tips:")
        print("   - Check if matches exist in this date range")
        print("   - Try expanding the date range")
        print("   - Verify the season is correct")
        return pd.DataFrame()  # Return empty DataFrame
    
    print("\n📥 Fetching match data...")
    matches_data = main.getMatchesData(match_urls=match_urls)
    
    # Check if matches_data is empty
    if not matches_data:
        print("\n⚠️  No match data retrieved")
        return pd.DataFrame()
    
    print(f"✅ Retrieved data for {len(matches_data)} matches")
    
    print("\n⚙️  Processing events...")
    events_ls = [main.createEventsDF(match) for match in matches_data]
    
    # Check if events_ls is empty
    if len(events_ls) == 0:
        print("\n⚠️  No events data to process")
        return pd.DataFrame()
    
    # Add EPV column
    print("📊 Calculating EPV values...")
    events_list = [main.addEpvToDataFrame(match) for match in events_ls]
    
    # Another critical check before concat
    if len(events_list) == 0:
        print("\n⚠️  No events data after EPV calculation")
        return pd.DataFrame()
    
    events_dfs = pd.concat(events_list)
    
    print("🔄 Aggregating data...")
    events_dfs['team'] = events_dfs['homeTeam'].where(events_dfs['h_a'] == 'h', events_dfs['awayTeam'])
    events_dfs['opponent'] = events_dfs['awayTeam'].where(events_dfs['h_a'] == 'h', events_dfs['homeTeam'])
    events_dfs['startDate'] = pd.to_datetime(events_dfs['startDate'])
    
    epv = events_dfs.groupby(["matchId", "team", 'opponent', 'startDate']).agg({
        "EPV": "sum"
    }).reset_index()
    
    epv['season'] = season_label
    epv['division'] = division
    
    # Team name mapping
    team_mapping = {
        'Nottingham Forest': 'Nottm Forest',
        'Manchester City': 'Man City',
        'Manchester United': 'Man United',
    }
    epv['team'] = epv['team'].replace(team_mapping)
    
    print(f"\n✅ Processed {len(epv)} EPV records")
    
    # Save to database
    print("\n💾 Saving EPV data to database...")
    save_epv_to_database(epv)
    
    return epv

# Main execution
if __name__ == "__main__":
    # Default values when running the script directly
    start_date = datetime(2025, 12, 1)
    end_date = datetime(2026, 1, 9)
    
    print("=" * 60)
    print("🏆 WhoScored EPV Data Scraper")
    print("=" * 60)
    
    epv_df = process_epv_data(
        start_date=start_date,
        end_date=end_date,
        competition='england-premier-league',
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