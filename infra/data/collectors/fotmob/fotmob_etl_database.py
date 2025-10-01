import json
import pandas as pd
import numpy as np
import os
import sqlite3
from pathlib import Path
from datetime import datetime

def process_json_files(folder_path):
    """
    Process all JSON files in a folder and extract shots, red cards, and match data.
    
    Args:
        folder_path (str): Path to the folder containing JSON files
    
    Returns:
        tuple: (shots_df, red_cards_df, matches_df) - Three consolidated DataFrames
    """
    
    # Initialize empty lists to store data
    all_shots = []
    all_red_cards = []
    all_matches = []
    
    # Get all JSON files in the folder
    json_files = list(Path(folder_path).glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return None, None, None
    
    print(f"Processing {len(json_files)} JSON files...")
    
    for file_path in json_files:
        try:
            print(f"Processing: {file_path.name}")
            
            # Load JSON data with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract match ID and team IDs
            match_id = data['matchFacts']['matchId']
            home_team = data['matchFacts']['poll']['oddspoll']['HomeTeamId']
            away_team = data['matchFacts']['poll']['oddspoll']['AwayTeamId']
            
            # Process shots data
            if 'shotmap' in data and 'shots' in data['shotmap']:
                shots = data['shotmap']['shots']
                for shot in shots:
                    shot['match_id'] = match_id
                    shot['home_team'] = home_team
                    shot['away_team'] = away_team
                all_shots.extend(shots)
            
            # Process red cards data
            if 'matchFacts' in data and 'events' in data['matchFacts']:
                events = data['matchFacts']['events']['events']
                
                for event in events:
                    if (event.get('type') == 'Card' and 
                        event.get('card') in ['Red', 'YellowRed']):
                        
                        is_home = event.get('isHome', False)
                        red_card = {
                            'match_id': match_id,
                            'team_id': home_team if is_home else away_team,
                            'time': event.get('time'),
                            'nameStr': event.get('nameStr'),
                            'type': 'Red Card',
                            'home_team': home_team,
                            'away_team': away_team
                        }
                        all_red_cards.append(red_card)
            
            # Process match data
            try:
                match_date = data['matchFacts']['infoBox']['Match Date']['utcTime']
                league = data['matchFacts']['infoBox']['Tournament']['id']
                
                home_goals = None
                away_goals = None
                ft_found = False
                for event in data['matchFacts']["events"]["events"]:
                    if event.get("halfStrShort") == "FT":
                        home_goals = event.get("homeScore", 0)
                        away_goals = event.get("awayScore", 0)
                        ft_found = True
                        break

                if ft_found:
                    match_info = {
                        'match_id': match_id,
                        'league_id': league,
                        'match_date': match_date,
                        'home_team': home_team,
                        'home_goals': home_goals,
                        'away_team': away_team,
                        'away_goals': away_goals
                    }
                    all_matches.append(match_info)
                else:
                    print(f"Warning: No FT event found for match {match_id}")
                    
            except KeyError as e:
                print(f"Warning: Missing match data key {e} in {file_path.name}")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            continue
    
    # Create DataFrames
    print("Creating DataFrames...")
    
    # Shots DataFrame
    if all_shots:
        shots_df = pd.DataFrame(all_shots)
        # Add side column (home or away)
        shots_df['side'] = shots_df.apply(
            lambda row: 'home' if row['teamId'] == row['home_team'] else 'away', 
            axis=1
        )
        # Select and reorder columns
        shots_columns = ["match_id", "teamId", "side", "min", "playerName", "eventType", "expectedGoals", "expectedGoalsOnTarget"]
        shots_df = shots_df[[col for col in shots_columns if col in shots_df.columns]]
        print(f"Created shots DataFrame with {len(shots_df)} rows")
    else:
        shots_df = pd.DataFrame(columns=["match_id", "teamId", "side", "min", "playerName", "eventType", "expectedGoals", "expectedGoalsOnTarget"])
        print("No shots data found")
    
    # Red Cards DataFrame
    if all_red_cards:
        red_cards_df = pd.DataFrame(all_red_cards)
        # Add side column (home or away)
        red_cards_df['side'] = red_cards_df.apply(
            lambda row: 'home' if row['team_id'] == row['home_team'] else 'away',
            axis=1
        )
        red_cards_df = red_cards_df[["match_id", "team_id", "side", "time", "nameStr", "type"]]
        print(f"Created red cards DataFrame with {len(red_cards_df)} rows")
    else:
        red_cards_df = pd.DataFrame(columns=["match_id", "team_id", "side", "time", "nameStr", "type"])
        print("No red card data found")
    
    # Matches DataFrame
    if all_matches:
        matches_df = pd.DataFrame(all_matches)
        # Convert match_date to yyyy-mm-dd format
        matches_df['match_date'] = pd.to_datetime(matches_df['match_date']).dt.strftime('%Y-%m-%d')
        matches_df = matches_df[["match_id", "league_id", "match_date", "home_team", "home_goals", "away_team", "away_goals"]]
        print(f"Created matches DataFrame with {len(matches_df)} rows")
    else:
        matches_df = pd.DataFrame(columns=["match_id", "league_id", "match_date", "home_team", "home_goals", "away_team", "away_goals"])
        print("No match data found")
    
    # Merge match_date into shots_df and red_cards_df
    if not matches_df.empty:
        if not shots_df.empty:
            shots_df = shots_df.merge(
                matches_df[['match_id', 'match_date']], 
                on='match_id', 
                how='left'
            )
            # Reorder columns to put match_date near the beginning
            cols = shots_df.columns.tolist()
            cols.remove('match_date')
            cols.insert(1, 'match_date')
            shots_df = shots_df[cols]
        
        if not red_cards_df.empty:
            red_cards_df = red_cards_df.merge(
                matches_df[['match_id', 'match_date']], 
                on='match_id', 
                how='left'
            )
            # Reorder columns to put match_date near the beginning
            cols = red_cards_df.columns.tolist()
            cols.remove('match_date')
            cols.insert(1, 'match_date')
            red_cards_df = red_cards_df[cols]
    
    return shots_df, red_cards_df, matches_df

def save_to_database(shots_df, red_cards_df, matches_df, season, league, db_name="infra/data/db/fotmob.db"):
    """
    Save the DataFrames to SQLite database tables with duplicate checking.
    
    Args:
        shots_df (pd.DataFrame): Shots data
        red_cards_df (pd.DataFrame): Red cards data  
        matches_df (pd.DataFrame): Matches data
        season (str): Season identifier
        league (str): League identifier
        db_name (str): Name of the SQLite database file
    """
    # Add season and league_id to each dataframe
    shots_df["season"] = season
    red_cards_df["season"] = season
    matches_df["season"] = season

    shots_df["league_id"] = league
    red_cards_df["league_id"] = league
    matches_df["league_id"] = league

    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    
    try:
        # Helper function to remove duplicates before inserting
        def insert_without_duplicates(df, table_name, conn, key_columns=None):
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Read existing data
                existing_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                
                # If key_columns specified, use those for duplicate checking
                # Otherwise use all columns
                if key_columns:
                    check_columns = key_columns
                else:
                    check_columns = df.columns.tolist()
                
                # Round numeric columns to avoid floating point precision issues
                df_check = df.copy()
                existing_check = existing_df.copy()
                
                for col in check_columns:
                    if col in df_check.columns and pd.api.types.is_numeric_dtype(df_check[col]):
                        df_check[col] = df_check[col].round(6)
                    if col in existing_check.columns and pd.api.types.is_numeric_dtype(existing_check[col]):
                        existing_check[col] = existing_check[col].round(6)
                
                # Create composite keys for comparison
                existing_keys = set(existing_check[check_columns].apply(lambda row: tuple(row), axis=1))
                new_rows_mask = ~df_check[check_columns].apply(lambda row: tuple(row) in existing_keys, axis=1)
                new_df = df[new_rows_mask]
                
                rows_before = len(df)
                rows_after = len(new_df)
                duplicates = rows_before - rows_after
                
                if duplicates > 0:
                    print(f"  Skipped {duplicates} duplicate rows in '{table_name}'")
                
                # Insert only new rows
                if len(new_df) > 0:
                    new_df.to_sql(table_name, conn, if_exists='append', index=False)
                    print(f"  Inserted {len(new_df)} new rows to '{table_name}' table")
                else:
                    print(f"  No new rows to insert in '{table_name}' table")
            else:
                # Table doesn't exist, insert all rows
                df.to_sql(table_name, conn, if_exists='append', index=False)
                print(f"  Created '{table_name}' table and inserted {len(df)} rows")
        
        # Insert data with duplicate checking
        print("\nProcessing shots table:")
        # For shots, use key columns that uniquely identify a shot (excluding xG values which may have precision issues)
        insert_without_duplicates(shots_df, 'shots', conn, 
                                 key_columns=['match_id', 'teamId', 'min', 'playerName', 'eventType', 'season', 'league_id'])
        
        print("\nProcessing red_cards table:")
        insert_without_duplicates(red_cards_df, 'red_cards', conn,
                                 key_columns=['match_id', 'team_id', 'time', 'nameStr', 'season', 'league_id'])
        
        print("\nProcessing matches table:")
        insert_without_duplicates(matches_df, 'matches', conn,
                                 key_columns=['match_id', 'season', 'league_id'])
        
        print(f"\n✓ Data successfully saved to {db_name}")
        
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        
    finally:
        conn.close()

# Main execution
if __name__ == "__main__":
    # Set the folder path containing your JSON files
    season = "2021-2022"
    league = "Premier_League"
    folder_path = r"infra/data/json" + "/" + league + "/" + season
    
    # Process all JSON files
    shots_df, red_cards_df, matches_df = process_json_files(folder_path)
    
    if shots_df is not None:
        # Save to SQLite database
        save_to_database(shots_df, red_cards_df, matches_df, season=season, league=league)
        
        # You can also access the DataFrames individually:
        # shots_df, red_cards_df, matches_df are now available for further processing
    else:
        print("No data processed. Please check your folder path and JSON files.")