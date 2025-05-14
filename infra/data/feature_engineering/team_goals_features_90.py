import pandas as pd
import sqlite3
import numpy as np

conn = sqlite3.connect(r'C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db')

df = pd.read_sql_query("SELECT * FROM fbref_match_summary", conn)

def apply_weighted_avg(col, match_date, match_red, decay_rate=0.015, time_window=90):
    # Create a mask for non-NaN values
    valid_mask = ~pd.isna(col)
    
    # If all values are NaN, return NaN
    if not valid_mask.any():
        return np.nan
    
    # Filter out NaN values
    valid_col = col[valid_mask].copy()  # Create a copy to avoid modifying original
    valid_dates = match_date[valid_mask]
    valid_red = match_red[valid_mask]
    
    # Get most recent date
    recent_date = max(valid_dates)
    
    # Create a time window mask (only include matches within time_window days)
    time_window_mask = (recent_date - valid_dates).dt.days <= time_window
    
    # If no matches in the time window, return NaN
    if not time_window_mask.any():
        return np.nan
    
    # Apply time window filter
    valid_col = valid_col[time_window_mask]
    valid_dates = valid_dates[time_window_mask]
    valid_red = valid_red[time_window_mask]
    
    # Calculate weights for matches within the time window
    match_weight = np.exp(-(recent_date - valid_dates).dt.days * decay_rate)
    
    # Reduce weight for matches with red cards (now using 0.3 instead of 0.5)
    match_weight = np.where(valid_red == 1, match_weight * 0.3, match_weight)
    
    # Ensure valid_col is numeric
    try:
        valid_col = pd.to_numeric(valid_col)
    except:
        # If conversion fails, return NaN
        return np.nan
    
    # Calculate weighted average using numpy to avoid pandas Series multiplication issues
    weighted_avg = np.sum(match_weight * valid_col.values) / np.sum(match_weight)

    return weighted_avg

df = df.drop_duplicates(subset=['match_url', 'team'])

df['match_date'] = pd.to_datetime(df['match_date'])
df['match_red'] = df["cards_red"].astype(int) + df["opp_cards_red"].astype(int)

# Define metrics to use
attack_metrics = ['goals', 'shots', 'shots_on_target', 'xg', 'npxg', 'touches_att_pen_area', 'touches_att_3rd', 'touches', 'pens_won', 'corner_kicks']
defense_metrics = ['opp_goals', 'opp_shots', 'opp_shots_on_target', 'opp_xg', 'opp_npxg', 'opp_touches_att_pen_area', 'opp_touches_att_3rd', 'opp_touches', 'opp_pens_won', 'opp_corner_kicks']

# Create a unique match identifier
df['match_id'] = df['match_url']  

# Process each team
team_metrics = []

for team_name in df['team'].unique():
    # Get all matches for this team
    team_matches = df[df['team'] == team_name].sort_values('match_date')
    
    # For each match, calculate weighted averages of previous matches
    for i, (idx, current_match) in enumerate(team_matches.iterrows()):
        if i > 0:  # Skip first match
            prev_matches = team_matches.iloc[:i]
            
            # Calculate all metrics in one go
            metrics_dict = {}
            
            # Attack metrics
            for metric in attack_metrics:
                weighted_avg = apply_weighted_avg(
                    prev_matches[metric],
                    prev_matches['match_date'],
                    prev_matches['match_red']
                )
                metrics_dict[f'weighted_attack_{metric}'] = weighted_avg
            
            # Defense metrics
            for metric in defense_metrics:
                weighted_avg = apply_weighted_avg(
                    prev_matches[metric],
                    prev_matches['match_date'],
                    prev_matches['match_red']
                )
                metrics_dict[f'weighted_defense_{metric}'] = weighted_avg
            
            # Add match info
            metrics_dict['team'] = team_name
            metrics_dict['match_id'] = current_match['match_id']
            metrics_dict['opp_team'] = current_match['opp_team']
            
            team_metrics.append(metrics_dict)

# Convert to dataframe
metrics_df = pd.DataFrame(team_metrics)

# When merging, keep all team-match combinations from the original data
final_df = df.merge(
    metrics_df,
    on=['team', 'match_id', 'opp_team'],
    how='left'
)

# Create opponent metrics dataframe
opp_metrics_df = metrics_df.rename(columns={
    'team': 'opp_team',
    'opp_team': 'team'
})
opp_metrics_df.columns = [f'opp_{col}' if col.startswith('weighted') else col for col in opp_metrics_df.columns]

# Merge opponent metrics
final_df = final_df.merge(
    opp_metrics_df,
    on=['team', 'match_id', 'opp_team'],
    how='left'
)

feature_cols = ["match_url", "match_date", "season", "team", "opp_team", "is_home", "goals", "opp_goals", "xg", "opp_xg",
                # attack metrics
                "weighted_attack_goals", "weighted_attack_shots", "weighted_attack_shots_on_target", "weighted_attack_xg", "weighted_attack_npxg", 
                "weighted_attack_touches_att_pen_area", "weighted_attack_touches_att_3rd", "weighted_attack_touches",
                "weighted_attack_pens_won", "weighted_attack_corner_kicks",
                # opposition defence metrics
                "opp_weighted_defense_opp_goals","opp_weighted_defense_opp_shots", "opp_weighted_defense_opp_shots_on_target", "opp_weighted_defense_opp_xg",
                "opp_weighted_defense_opp_npxg", "opp_weighted_defense_opp_touches_att_pen_area", "opp_weighted_defense_opp_touches_att_3rd",
                "opp_weighted_defense_opp_touches", "opp_weighted_defense_opp_pens_won", "opp_weighted_defense_opp_corner_kicks"]

# Create a features dataframe with only the selected columns
features_df = final_df[feature_cols].copy()

# Write to a new table in your SQLite database
features_df.to_sql('fbref_team_goals_features_015_90', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print(f"Feature table created with {len(features_df)} rows and {len(feature_cols)} columns")