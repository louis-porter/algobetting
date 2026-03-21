import pandas as pd
import understatapi

def scrape_understat_shots(league, season):
    client = understatapi.UnderstatClient()
    league_data = client.league(league=league).get_match_data(season=season)
    league_data = [match for match in league_data if match.get('isResult') == True]
    
    all_shots = []
    all_matches = []

    for match in league_data:
        match_id = match.get('id')
        home_team_id = match['h']['id']
        away_team_id = match['a']['id']
        home_team_name = match['h']['title']
        away_team_name = match['a']['title']

        # Build understat_matches row
        all_matches.append({
            'match_id': match_id,
            'match_date': match.get('datetime'),
            'home_team_id': home_team_id,
            'home_team_name': home_team_name,
            'away_team_id': away_team_id,
            'away_team_name': away_team_name,
            'home_goals': match['goals']['h'],
            'away_goals': match['goals']['a'],
            'home_xG': match['xG']['h'],
            'away_xG': match['xG']['a'],
        })
        
        try:
            shot_data = client.match(match=match_id).get_shot_data()
            
            for side in ['h', 'a']:
                shots = shot_data.get(side, [])
                for shot in shots:
                    shot['match_id'] = match_id
                    shot['side'] = side
                    shot['home_team_id'] = home_team_id
                    shot['home_team_name'] = home_team_name
                    shot['away_team_id'] = away_team_id
                    shot['away_team_name'] = away_team_name
                all_shots.extend(shots)
                
        except Exception as e:
            print(f"Warning: Could not fetch shot data for match {match_id}: {e}")
            continue
    
    # --- Shots DataFrame ---
    if not all_shots:
        print("No shot data found")
        shots_df = pd.DataFrame()
    else:
        shots_df = pd.DataFrame(all_shots)
        shots_df = shots_df[shots_df['situation'] != 'Penalty']
        
        shots_df = shots_df.rename(columns={
            'id': 'shot_id',
            'minute': 'min',
            'result': 'eventType',
            'X': 'x',
            'Y': 'y',
            'xG': 'expectedGoals',
            'player': 'playerName',
            'player_id': 'player_id',
            'situation': 'situation',
            'type': 'shot_type',
        })
        
        desired_cols = [
            'match_id', 'shot_id', 'side', 'min', 'playerName', 'player_id',
            'eventType', 'expectedGoals', 'situation', 'shot_type',
            'x', 'y',
            'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name'
        ]
        shots_df = shots_df[[col for col in desired_cols if col in shots_df.columns]]
        print(f"Created understat shots DataFrame with {len(shots_df)} rows")

    # --- Matches DataFrame ---
    if not all_matches:
        print("No match data found")
        matches_df = pd.DataFrame()
    else:
        matches_df = pd.DataFrame(all_matches)
        matches_df['match_date'] = pd.to_datetime(matches_df['match_date']).dt.strftime('%Y-%m-%d')
        matches_df[['home_goals', 'away_goals']] = matches_df[['home_goals', 'away_goals']].astype(int)
        matches_df[['home_xG', 'away_xG']] = matches_df[['home_xG', 'away_xG']].astype(float).round(4)
        print(f"Created understat matches DataFrame with {len(matches_df)} rows")

    return shots_df, matches_df


def save_understat_to_database(shots_df, matches_df, season, league, db_name="infra/data/db/fotmob.db"):
    import sqlite3

    def insert_without_duplicates(df, table_name, conn, key_columns=None):
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            existing_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            check_columns = key_columns if key_columns else df.columns.tolist()
            
            df_check = df.copy()
            existing_check = existing_df.copy()
            
            for col in check_columns:
                if col in df_check.columns and pd.api.types.is_numeric_dtype(df_check[col]):
                    df_check[col] = df_check[col].round(6)
                if col in existing_check.columns and pd.api.types.is_numeric_dtype(existing_check[col]):
                    existing_check[col] = existing_check[col].round(6)
            
            existing_keys = set(existing_check[check_columns].apply(lambda row: tuple(row), axis=1))
            new_rows_mask = ~df_check[check_columns].apply(lambda row: tuple(row) in existing_keys, axis=1)
            new_df = df[new_rows_mask]
            
            duplicates = len(df) - len(new_df)
            if duplicates > 0:
                print(f"  Skipped {duplicates} duplicate rows in '{table_name}'")
            
            if len(new_df) > 0:
                new_df.to_sql(table_name, conn, if_exists='append', index=False)
                print(f"  Inserted {len(new_df)} new rows to '{table_name}' table")
            else:
                print(f"  No new rows to insert in '{table_name}' table")
        else:
            df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"  Created '{table_name}' table and inserted {len(df)} rows")

    for df, label in [(shots_df, 'shots'), (matches_df, 'matches')]:
        if df is None or df.empty:
            print(f"No Understat {label} data to save.")
            return

    shots_df = shots_df.copy()
    matches_df = matches_df.copy()
    for df in [shots_df, matches_df]:
        df['season'] = season
        df['league_id'] = league

    conn = sqlite3.connect(db_name)
    try:
        print("\nProcessing understat_shots table:")
        insert_without_duplicates(shots_df, 'understat_shots', conn,
                                  key_columns=['match_id', 'shot_id', 'season', 'league_id'])

        print("\nProcessing understat_matches table:")
        insert_without_duplicates(matches_df, 'understat_matches', conn,
                                  key_columns=['match_id', 'season', 'league_id'])

        print(f"\n✓ Understat data successfully saved to {db_name}")
    except Exception as e:
        print(f"Error saving Understat data to database: {str(e)}")
    finally:
        conn.close()


def main_understat(season, league):
    shots_df, matches_df = scrape_understat_shots(league=league, season=season)
    save_understat_to_database(shots_df, matches_df, season=season, league=league)


if __name__ == "__main__":
    season = "2025"
    league = "EPL"
    main_understat(season, league)