import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def apply_weighted_avg(col, match_date, match_red, decay_rate=0.005, time_window=365, debug=False):
    """
    Calculate weighted average with optional debug information.
    Setting debug=True provides detailed information about the calculation.
    """
    # Create a mask for non-NaN values
    valid_mask = ~pd.isna(col)
    
    if debug:
        print(f"Total values: {len(col)}, Valid values: {sum(valid_mask)}")
    
    # If all values are NaN, return NaN
    if not valid_mask.any():
        if debug:
            print("All values are NaN, returning NaN")
        return np.nan
    
    # Filter out NaN values
    valid_col = col[valid_mask].copy()  # Create a copy to avoid modifying original
    valid_dates = match_date[valid_mask]
    valid_red = match_red[valid_mask]
    
    # Get most recent date
    recent_date = max(valid_dates)
    
    if debug:
        print(f"Most recent date: {recent_date}")
        print("\nValues before time window filter:")
        for i, (val, date, red) in enumerate(zip(valid_col, valid_dates, valid_red)):
            days_ago = (recent_date - date).days
            print(f"  {i+1}. Value: {val}, Date: {date}, Days ago: {days_ago}, Red card: {red}")
    
    # Create a time window mask (only include matches within time_window days)
    time_window_mask = (recent_date - valid_dates).dt.days <= time_window
    
    # If no matches in the time window, return NaN
    if not time_window_mask.any():
        if debug:
            print("No matches within time window, returning NaN")
        return np.nan
    
    # Apply time window filter
    valid_col = valid_col[time_window_mask]
    valid_dates = valid_dates[time_window_mask]
    valid_red = valid_red[time_window_mask]
    
    if debug:
        print(f"\nValues after time window filter ({time_window} days):")
        for i, (val, date, red) in enumerate(zip(valid_col, valid_dates, valid_red)):
            days_ago = (recent_date - date).days
            print(f"  {i+1}. Value: {val}, Date: {date}, Days ago: {days_ago}, Red card: {red}")
    
    # Calculate weights for matches within the time window
    days_diff = (recent_date - valid_dates).dt.days
    match_weight = np.exp(-days_diff * decay_rate)
    
    # Reduce weight for matches with red cards 
    original_weights = match_weight.copy()
    match_weight = np.where(valid_red == 1, match_weight * 0.25, match_weight)
    
    if debug:
        print("\nWeight calculations:")
        for i, (val, date, red, orig_w, w) in enumerate(zip(valid_col, valid_dates, valid_red, original_weights, match_weight)):
            days_ago = (recent_date - date).days
            weight_calc = f"exp(-{days_ago} * {decay_rate}) = {orig_w:.4f}"
            if red == 1:
                weight_calc += f" * 0.25 = {w:.4f}"
            print(f"  {i+1}. Value: {val}, Date: {date}, Days ago: {days_ago}, Red card: {red}")
            print(f"     Weight calculation: {weight_calc}")
    
    # Ensure valid_col is numeric
    try:
        valid_col_numeric = pd.to_numeric(valid_col)
        if debug:
            print("\nConverted values to numeric:")
            for i, (orig, num) in enumerate(zip(valid_col, valid_col_numeric)):
                print(f"  {i+1}. Original: '{orig}' → Numeric: {num}")
        valid_col = valid_col_numeric
    except Exception as e:
        if debug:
            print(f"Numeric conversion error: {e}")
            print("Values that failed conversion:")
            for i, val in enumerate(valid_col):
                print(f"  {i+1}. '{val}' (type: {type(val)})")
        return np.nan
    
    # Calculate weighted average using numpy to avoid pandas Series multiplication issues
    weighted_avg = np.sum(match_weight * valid_col.values) / np.sum(match_weight)
    
    if debug:
        print("\nWeighted average calculation:")
        sum_weighted_values = 0
        total_weights = 0
        for i, (val, w) in enumerate(zip(valid_col, match_weight)):
            weighted_val = val * w
            sum_weighted_values += weighted_val
            total_weights += w
            print(f"  {i+1}. Value: {val} × Weight: {w:.4f} = {weighted_val:.4f}")
        print(f"  Sum of weighted values: {sum_weighted_values:.4f}")
        print(f"  Sum of weights: {total_weights:.4f}")
        print(f"  Weighted average: {sum_weighted_values:.4f} / {total_weights:.4f} = {weighted_avg:.4f}")

    return weighted_avg

def debug_arsenal_weighted_goals():
    """
    Debug the weighted goals calculation for Arsenal specifically
    """
    # Connect to database
    db_path = r"C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db"  # Update this path
    table_name = "fbref_match_summary"  # Update this table name
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get Arsenal match data
        query = f"""
            SELECT 
                team, match_date, opp_team, goals, opp_goals, cards_red,
                match_url, is_home
            FROM {table_name}
            WHERE team = 'Arsenal'
            ORDER BY match_date
        """
        
        arsenal_df = pd.read_sql_query(query, conn)
        
        # Convert match_date to datetime
        arsenal_df['match_date'] = pd.to_datetime(arsenal_df['match_date'])
        
        # Convert cards_red to numeric
        arsenal_df['cards_red'] = pd.to_numeric(arsenal_df['cards_red'], errors='coerce').fillna(0)
        
        # Convert goals and opp_goals to numeric
        arsenal_df['goals'] = pd.to_numeric(arsenal_df['goals'], errors='coerce')
        arsenal_df['opp_goals'] = pd.to_numeric(arsenal_df['opp_goals'], errors='coerce')
        
        print(f"Found {len(arsenal_df)} Arsenal matches")
        print("\nMatch data preview:")
        print(arsenal_df[['match_date', 'opp_team', 'goals', 'opp_goals', 'cards_red']].head(2))
        
        # For each match, calculate the weighted average using only prior matches
        print("\n\nCalculating weighted opp_goals for each match using prior data:")
        for i in range(1, min(2, len(arsenal_df))):  # Start from the second match, look at first 10 max
            match = arsenal_df.iloc[i]
            prior_matches = arsenal_df.iloc[:i]
            
            print(f"\n\n{'='*80}")
            print(f"MATCH {i+1}: Arsenal vs {match['opp_team']} on {match['match_date'].strftime('%Y-%m-%d')}")
            print(f"{'='*80}")
            
            # Get prior data
            prior_opp_goals = prior_matches['opp_goals']
            prior_dates = prior_matches['match_date']
            prior_red_cards = prior_matches['cards_red']
            
            print(f"\nCalculating weighted average of opp_goals using {len(prior_matches)} prior matches:")
            
            # Calculate weighted average with debugging output
            weighted_avg = apply_weighted_avg(
                prior_opp_goals, 
                prior_dates, 
                prior_red_cards,
                decay_rate=0.005,  # Default value
                time_window=365,   # Default value
                debug=True
            )
            
            print(f"\nFinal weighted average opp_goals: {weighted_avg:.4f}")
            print(f"Actual opp_goals in current match: {match['opp_goals']}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


debug_arsenal_weighted_goals()