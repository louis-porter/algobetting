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
                print(f" Skipped {duplicates} duplicate rows in '{table_name}'")
            
            # Insert only new rows
            if len(new_df) > 0:
                new_df.to_sql(table_name, conn, if_exists='append', index=False)
                print(f" Inserted {len(new_df)} new rows to '{table_name}' table")
            else:
                print(f" No new rows to insert in '{table_name}' table")
        else:
            # Table doesn't exist, insert all rows
            epv_df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f" Created '{table_name}' table and inserted {len(epv_df)} rows")
        
        print(f"\n✓ EPV data successfully saved to {db_name}")
        
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
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
    print(f"Processing EPV data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    league_urls = main.getLeagueUrls()
    match_urls = main.getMatchUrls(comp_urls=league_urls, competition=competition, season=season)
    
    # Convert dates for ALL matches
    for match in match_urls:
        match['date_dt'] = datetime.strptime(match['date'], '%A, %b %d %Y')
    
    # Filter matches by date range
    match_urls = [match for match in match_urls if start_date <= match['date_dt'] <= end_date]
    
    print(f"Found {len(match_urls)} matches in date range")
    
    matches_data = main.getMatchesData(match_urls=match_urls)
    events_ls = [main.createEventsDF(match) for match in matches_data]
    
    # Add EPV column
    events_list = [main.addEpvToDataFrame(match) for match in events_ls]
    events_dfs = pd.concat(events_list)
    
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
    
    # Save to database
    print("\nSaving EPV data to database...")
    save_epv_to_database(epv)
    
    return epv

# Main execution
if __name__ == "__main__":
    # Default values when running the script directly
    start_date = datetime(2025, 8, 1)
    end_date = datetime(2025, 10, 10)
    
    epv_df = process_epv_data(
        start_date=start_date,
        end_date=end_date,
        competition='england-premier-league',
        season='2025/2026',
        season_label='2025-2026',
        division='Premier_League'
    )