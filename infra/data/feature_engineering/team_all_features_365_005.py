import pandas as pd
import sqlite3
import numpy as np

conn = sqlite3.connect(r'C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db')

df = pd.read_sql_query("SELECT * FROM fbref_match_summary_v2", conn)

def apply_weighted_avg(col, match_date, match_red, current_match_date, decay_rate=0.005, time_window=365, min_games=5):
    # Create a mask for non-NaN values
    valid_mask = ~pd.isna(col)
    
    # If all values are NaN, return NaN
    if not valid_mask.any():
        return np.nan
    
    # Filter out NaN values
    valid_col = col[valid_mask].copy()
    valid_dates = match_date[valid_mask]
    valid_red = match_red[valid_mask]
    
    # Create a time window mask (only include matches within time_window days BEFORE current match)
    time_window_mask = (current_match_date - valid_dates).dt.days <= time_window
    
    # If no matches in the time window, return NaN
    if not time_window_mask.any():
        return np.nan
    
    # Apply time window filter
    valid_col = valid_col[time_window_mask]
    valid_dates = valid_dates[time_window_mask]
    valid_red = valid_red[time_window_mask]
    
    # Check if we have minimum required games
    if len(valid_col) < min_games:
        return np.nan
    
    # Calculate weights for matches within the time window (relative to current match date)
    match_weight = np.exp(-(current_match_date - valid_dates).dt.days * decay_rate)
    
    # Reduce weight for matches with red cards
    match_weight = np.where(valid_red == 1, match_weight * 0.3, match_weight)
    
    # Ensure valid_col is numeric
    try:
        valid_col = pd.to_numeric(valid_col)
    except:
        return np.nan
    
    # Calculate weighted average
    weighted_avg = np.sum(match_weight * valid_col.values) / np.sum(match_weight)
    return weighted_avg

# Clean and prepare data
df = df.drop_duplicates(subset=['match_url', 'team'])
df['match_date'] = pd.to_datetime(df['match_date'])
df['match_red'] = df["summary_cards_red"].astype(int) + df["opp_summary_cards_red"].astype(int)
df['match_id'] = df['match_url']

# Automatically identify all numeric columns to process
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove columns we don't want to process
exclude_cols = ['match_red', 'summary_cards_red', 'opp_summary_cards_red', 'is_home']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# Separate team stats (without opp_ prefix) and opponent stats (with opp_ prefix)
team_stats = [col for col in numeric_cols if not col.startswith('opp_')]
opp_stats = [col for col in numeric_cols if col.startswith('opp_')]

print(f"Processing {len(team_stats)} team stats and {len(opp_stats)} opponent stats")

# Process each team
team_metrics = []

for team_name in df['team'].unique():
    team_matches = df[df['team'] == team_name].sort_values('match_date')
    
    for i, (idx, current_match) in enumerate(team_matches.iterrows()):
        if i > 0:  # Skip first match (no previous data)
            prev_matches = team_matches.iloc[:i]
            
            metrics_dict = {}
            
            # Calculate rolling averages for all team stats
            for stat in team_stats:
                weighted_avg = apply_weighted_avg(
                    prev_matches[stat],
                    prev_matches['match_date'],
                    prev_matches['match_red'],
                    current_match['match_date']  # Pass current match date as reference
                )
                metrics_dict[f'team_rolling_{stat}'] = weighted_avg
            
            # Calculate rolling averages for all opponent stats (defensive perspective)
            for stat in opp_stats:
                weighted_avg = apply_weighted_avg(
                    prev_matches[stat],
                    prev_matches['match_date'],
                    prev_matches['match_red'],
                    current_match['match_date']  # Pass current match date as reference
                )
                # Remove 'opp_' prefix and add 'team_rolling_conceded_' prefix
                clean_stat = stat.replace('opp_', '')
                metrics_dict[f'team_rolling_conceded_{clean_stat}'] = weighted_avg
            
            # Add match info
            metrics_dict['team'] = team_name
            metrics_dict['match_id'] = current_match['match_id']
            metrics_dict['opp_team'] = current_match['opp_team']
            
            team_metrics.append(metrics_dict)

# Convert to dataframe
metrics_df = pd.DataFrame(team_metrics)

# Merge with original data
final_df = df.merge(
    metrics_df,
    on=['team', 'match_id', 'opp_team'],
    how='left'
)

# Create opponent metrics by swapping team/opp_team
opp_metrics_df = metrics_df.copy()
opp_metrics_df = opp_metrics_df.rename(columns={
    'team': 'opp_team',
    'opp_team': 'team'
})

# Rename all rolling columns to have opp_ prefix
rolling_cols = [col for col in opp_metrics_df.columns if col.startswith('team_rolling_')]
rename_dict = {col: f'opp_{col}' for col in rolling_cols}
opp_metrics_df = opp_metrics_df.rename(columns=rename_dict)

# Merge opponent metrics
final_df = final_df.merge(
    opp_metrics_df,
    on=['team', 'match_id', 'opp_team'],
    how='left'
)

# Select key columns for the final features table
key_cols = ['match_url', 'match_date', 'season', 'division', 'team', 'opp_team', 'is_home']
rolling_cols = [col for col in final_df.columns if 'rolling' in col]

feature_cols = key_cols + rolling_cols
features_df = final_df[feature_cols].copy()

# Write to database
features_df.to_sql('team_all_features_365_005', conn, if_exists='replace', index=False)

conn.close()

print(f"Feature table created with {len(features_df)} rows and {len(feature_cols)} columns")
print(f"Rolling feature columns: {len(rolling_cols)}")
print("\nSample rolling columns:")
for col in rolling_cols[:10]:
    print(f"  {col}")