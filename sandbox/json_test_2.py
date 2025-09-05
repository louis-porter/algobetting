import json
import pandas as pd
import numpy as np
import os
from pathlib import Path

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
            
            # Extract match ID
            match_id = data['matchFacts']['matchId']
            
            # Process shots data
            if 'shotmap' in data and 'shots' in data['shotmap']:
                shots = data['shotmap']['shots']
                for shot in shots:
                    shot['match_id'] = match_id
                all_shots.extend(shots)
            
            # Process red cards data
            if 'matchFacts' in data and 'events' in data['matchFacts']:
                events = data['matchFacts']['events']['events']
                home_team = data['matchFacts']['poll']['oddspoll']['HomeTeamId']
                away_team = data['matchFacts']['poll']['oddspoll']['AwayTeamId']
                
                for event in events:
                    if (event.get('type') == 'Card' and 
                        event.get('card') in ['Red', 'YellowRed']):
                        
                        red_card = {
                            'match_id': match_id,
                            'team_id': home_team if event.get('isHome', False) else away_team,
                            'time': event.get('time'),
                            'nameStr': event.get('nameStr'),
                            'type': 'Red Card'
                        }
                        all_red_cards.append(red_card)
            
            # Process match data
            try:
                match_date = data['matchFacts']['infoBox']['Match Date']['utcTime']
                home_team = data['matchFacts']['poll']['oddspoll']['HomeTeamId']
                away_team = data['matchFacts']['poll']['oddspoll']['AwayTeamId']
                league = data['matchFacts']['infoBox']['Tournament']['id']
                
                home_goals = None
                away_goals = None
                ft_found = False
                for event in data['matchFacts']["events"]["events"]:
                    if event.get("halfStrShort") == "FT":
                        home_goals = event.get("homeScore", 0)  # Default to 0 if None
                        away_goals = event.get("awayScore", 0)  # Default to 0 if None
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
        # Select and reorder columns
        shots_columns = ["match_id", "teamId", "min", "playerName", "eventType", "expectedGoals", "expectedGoalsOnTarget"]
        shots_df = shots_df[[col for col in shots_columns if col in shots_df.columns]]
        print(f"Created shots DataFrame with {len(shots_df)} rows")
    else:
        shots_df = pd.DataFrame(columns=["match_id", "teamId", "min", "playerName", "eventType", "expectedGoals", "expectedGoalsOnTarget"])
        print("No shots data found")
    
    # Red Cards DataFrame
    if all_red_cards:
        red_cards_df = pd.DataFrame(all_red_cards)
        red_cards_df = red_cards_df[["match_id", "team_id", "time", "nameStr", "type"]]
        print(f"Created red cards DataFrame with {len(red_cards_df)} rows")
    else:
        red_cards_df = pd.DataFrame(columns=["match_id", "team_id", "time", "nameStr", "type"])
        print("No red card data found")
    
    # Matches DataFrame
    if all_matches:
        matches_df = pd.DataFrame(all_matches)
        matches_df = matches_df[["match_id", "league_id", "match_date", "home_team", "home_goals", "away_team", "away_goals"]]
        print(f"Created matches DataFrame with {len(matches_df)} rows")
    else:
        matches_df = pd.DataFrame(columns=["match_id", "league_id", "match_date", "home_team", "home_goals", "away_team", "away_goals"])
        print("No match data found")
    
    return shots_df, red_cards_df, matches_df

def save_dataframes(shots_df, red_cards_df, matches_df, season, output_folder="output"):
    """
    Save the DataFrames to CSV files.
    
    Args:
        shots_df (pd.DataFrame): Shots data
        red_cards_df (pd.DataFrame): Red cards data  
        matches_df (pd.DataFrame): Matches data
        output_folder (str): Folder to save CSV files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    shots_df["season"] = season
    red_cards_df["season"] = season
    matches_df["season"] = season


    # Save to CSV
    shots_df.to_csv(f"{output_folder}/shots_data.csv", index=False)
    red_cards_df.to_csv(f"{output_folder}/red_cards_data.csv", index=False)
    matches_df.to_csv(f"{output_folder}/matches_data.csv", index=False)
    
    print(f"DataFrames saved to {output_folder}/ folder")

# Main execution
if __name__ == "__main__":
    # Set the folder path containing your JSON files
    season = "2023-2024"
    folder_path = r"data\Premier_League" + "\\" + season
    
    # Process all JSON files
    shots_df, red_cards_df, matches_df = process_json_files(folder_path)
    
    if shots_df is not None:
        # Save to CSV files
        save_dataframes(shots_df, red_cards_df, matches_df, season=season)
        
        # You can also access the DataFrames individually:
        # shots_df, red_cards_df, matches_df are now available for further processing
    else:
        print("No data processed. Please check your folder path and JSON files.")