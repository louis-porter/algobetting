# =============================================================================
# src/data_utils.py
# =============================================================================

import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import poisson
from itertools import product
from typing import Dict, List, Tuple, Optional, Union

def load_football_data(db_path: str, league: Union[str, List[str]], season: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load match, shot, red card, and EPV data from SQLite database
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database
    league : str or list
        League identifier(s) (e.g., 'Premier_League')
    season : str
        Season identifier (e.g., '2023-2024')
    
    Returns:
    --------
    match_df, shot_df, red_df, epv_df : tuple of DataFrames
    """
    conn = sqlite3.connect(db_path)

    if isinstance(league, str):
        leagues = [league]
    else:
        leagues = league

    league_placeholders = ','.join(['?' for _ in leagues])
    
    # Load match data
    match_df = pd.read_sql_query(f"""
        SELECT 
            match_id,
            match_date,
            league_id,
            season,
            home.team_name as home_team,
            home_goals,
            away.team_name as away_team,
            away_goals
        FROM matches fmd
            JOIN team_id_mapping home ON home.team_id = fmd.home_team
            JOIN team_id_mapping away ON away.team_id = fmd.away_team
        WHERE
            league_id IN ({league_placeholders})
            AND season = ?
    """, conn, params=leagues + [season])
    
    # Load shot data
    shot_df = pd.read_sql_query(f"""
        SELECT 
            match_id,
            match_date,
            league_id,
            season,
            team.team_name as team,
            side,
            expectedGoals,
            expectedGoalsOnTarget
        FROM shots fsd
            JOIN team_id_mapping team ON team.team_id = fsd.teamId
        WHERE
            league_id IN ({league_placeholders})
            AND season = ?
    """, conn, params=leagues + [season])
    
    # Load red card data
    red_df = pd.read_sql_query(f"""
        SELECT 
            match_id,
            match_date,
            league_id,
            season,
            team.team_name as team,
            side,
            time,
            type
        FROM red_cards frcd
            JOIN team_id_mapping team ON team.team_id = frcd.team_id
        WHERE
            league_id IN ({league_placeholders})
            AND season = ?
    """, conn, params=leagues + [season])

    # Load EPV data
    epv_df = pd.read_sql_query(f"""
        SELECT DISTINCT
            red.match_id,
            red.match_date,
            epv.team,
            epv.EPV,
            epv.season,
            epv.division as league_id
        FROM shots red
        JOIN team_id_mapping team ON team.team_id = red.teamId
        JOIN epv ON epv.team = team.team_name AND red.match_date = DATE(epv.startDate)                  
        WHERE
            division IN ({league_placeholders})
            AND epv.season = ?
    """, conn, params=leagues + [season])

    conn.close()
    
    # Add days_ago calculation for all dataframes
    for df in [match_df, shot_df, red_df, epv_df]:
        df["days_ago"] = (pd.to_datetime(df["match_date"]).max() - pd.to_datetime(df["match_date"])).dt.days
        df["match_date"] = pd.to_datetime(df["match_date"])
    
    return match_df, shot_df, red_df, epv_df

def poisson_binomial_pmf(k: int, p_values: np.ndarray) -> float:
    """
    Calculate PMF of Poisson-Binomial distribution for k successes
    given array of success probabilities p_values
    
    Uses dynamic programming approach for efficiency
    """
    n = len(p_values)
    if k > n or k < 0:
        return 0.0
    
    # Dynamic programming table
    # dp[i][j] = probability of exactly j successes using first i trials
    dp = np.zeros((n + 1, k + 1))
    dp[0][0] = 1.0  # Base case: 0 trials, 0 successes
    
    for i in range(1, n + 1):
        p = p_values[i - 1]
        dp[i][0] = dp[i - 1][0] * (1 - p)  # 0 successes
        
        for j in range(1, min(i, k) + 1):
            # j successes = (j successes from first i-1, no success on trial i) +
            #               (j-1 successes from first i-1, success on trial i)
            dp[i][j] = dp[i - 1][j] * (1 - p) + dp[i - 1][j - 1] * p
    
    return dp[n][k]

def simulate_shots_poisson_binomial(xg_values: np.ndarray, max_goals: Optional[int] = None) -> Dict[int, float]:
    """
    Simulate goal probabilities using Poisson-Binomial distribution
    
    Args:
        xg_values: array of xG values for individual shots
        max_goals: maximum number of goals to consider (default: number of shots)
    
    Returns:
        dict with goals as keys and probabilities as values
    """
    if len(xg_values) == 0:
        return {0: 1.0}
    
    if max_goals is None:
        max_goals = len(xg_values)
    
    goal_probs = {}
    for goals in range(max_goals + 1):
        prob = poisson_binomial_pmf(goals, xg_values)
        if prob > 1e-10:  # Only keep non-negligible probabilities
            goal_probs[goals] = prob
    
    return goal_probs

def simulate_game_poisson_binomial(home_xg_shots: np.ndarray, away_xg_shots: np.ndarray, 
                                 max_goals: int = 9) -> List[Dict]:
    """
    Simulate a game using Poisson-Binomial distribution for each team
    
    Args:
        home_xg_shots: array of xG values for home team shots
        away_xg_shots: array of xG values for away team shots
        max_goals: maximum goals to consider for each team
    
    Returns:
        list of dicts with home_goals, away_goals, and probability
    """
    home_probs = simulate_shots_poisson_binomial(home_xg_shots, max_goals)
    away_probs = simulate_shots_poisson_binomial(away_xg_shots, max_goals)
    
    game_probs = []
    for (h_goals, h_prob), (a_goals, a_prob) in product(home_probs.items(), away_probs.items()):
        combined_prob = h_prob * a_prob
        if combined_prob > 1e-6:  # Filter very small probabilities
            game_probs.append({
                'home_goals': h_goals,
                'away_goals': a_goals,
                'probability': combined_prob
            })
    
    return game_probs

def calculate_red_card_penalty(red_cards_df: pd.DataFrame, match_id: str) -> float:
    """
    Calculate red card penalty for a specific match based on earliest red card timing
    
    Parameters:
    -----------
    red_cards_df : DataFrame
        Red card data
    match_id : str
        Match identifier
        
    Returns:
    --------
    float : penalty multiplier (lower = more penalty)
    """
    match_red_cards = red_cards_df[red_cards_df['match_id'] == match_id]
    
    if match_red_cards.empty:
        return 1.0  # No penalty if no red cards
    
    # Get the earliest red card minute for this match
    earliest_red_minute = match_red_cards['time'].min()
    
    # Apply penalty based on when the red card occurred
    if earliest_red_minute > 80:
        return 0.85
    elif earliest_red_minute > 70:
        return 0.75
    elif earliest_red_minute > 60:
        return 0.65
    elif earliest_red_minute > 45:
        return 0.5
    elif earliest_red_minute > 30:
        return 0.35
    elif earliest_red_minute > 15:
        return 0.2
    else:  # Red card in first 15 minutes
        return 0.05

def create_weighted_scoreline_data(match_df: pd.DataFrame, 
                                 shot_df: pd.DataFrame,
                                 red_df: pd.DataFrame,
                                 epv_df: pd.DataFrame,
                                 max_goals: int = 9,
                                 min_prob_threshold: float = 0.001,
                                 decay_rate: float = 0.001,
                                 goals_weight: float = 0.2,
                                 xg_weight: float = 0.45,
                                 psxg_weight: float = 0.25,
                                 epv_weight: float = 0.1) -> pd.DataFrame:
    """
    Create expanded dataset with all possible scorelines using Poisson-Binomial 
    and Poisson distributions with sophisticated weighting
    
    Parameters:
    -----------
    match_df : DataFrame
        Match results data
    shot_df : DataFrame
        Shot-level data with xG and psxG
    red_df : DataFrame
        Red card data
    epv_df : DataFrame
        EPV data
    max_goals : int
        Maximum goals to consider per team
    min_prob_threshold : float
        Minimum probability threshold for including scorelines
    decay_rate : float
        Time decay rate for match recency
    goals_weight : float
        Weight boost for actual scorelines
    xg_weight : float
        Weight for xG-based probabilities
    psxg_weight : float
        Weight for psxG-based probabilities
    epv_weight : float
        Weight for EPV-based probabilities
        
    Returns:
    --------
    pd.DataFrame : Expanded dataset with weighted scorelines
    """
    expanded_data = []
    
    for idx, row in match_df.iterrows():
        match_id = row['match_id']
        
        # Get shot data for this match
        match_shots = shot_df[shot_df["match_id"] == match_id]
        
        if match_shots.empty:
            print(f"Warning: No shot data found for match_id {match_id}")
            continue
            
        home_xg_shots = match_shots[match_shots['side'] == 'home']['expectedGoals'].values
        away_xg_shots = match_shots[match_shots['side'] == 'away']['expectedGoals'].values
        home_psxg_shots = match_shots[match_shots['side'] == 'home']['expectedGoalsOnTarget'].values
        away_psxg_shots = match_shots[match_shots['side'] == 'away']['expectedGoalsOnTarget'].values

        # Get EPV data for this match
        match_epv = epv_df[epv_df["match_id"] == match_id]
        home_epv = match_epv[match_epv['team'] == row['home_team']]['EPV'].values
        away_epv = match_epv[match_epv['team'] == row['away_team']]['EPV'].values
        
        home_total_epv = home_epv[0] if len(home_epv) > 0 else 0
        away_total_epv = away_epv[0] if len(away_epv) > 0 else 0

        # Calculate red card penalty
        red_card_penalty = calculate_red_card_penalty(red_df, match_id)
        
        # Calculate total xG for Poisson approach
        home_total_xg = home_xg_shots.sum()
        away_total_xg = away_xg_shots.sum()
        
        # Generate scorelines with Poisson-Binomial (individual shots)
        xg_game_probs_pb = simulate_game_poisson_binomial(home_xg_shots, away_xg_shots, max_goals)
        psxg_game_probs_pb = simulate_game_poisson_binomial(home_psxg_shots, away_psxg_shots, max_goals)
        
        # Generate scorelines with regular Poisson (total xG)
        home_total_xg_probs = {i: poisson.pmf(i, home_total_xg) for i in range(max_goals + 1)}
        away_total_xg_probs = {i: poisson.pmf(i, away_total_xg) for i in range(max_goals + 1)}
        
        # Generate scorelines with Poisson (EPV)
        home_epv_probs = {i: poisson.pmf(i, home_total_epv) for i in range(max_goals + 1)}
        away_epv_probs = {i: poisson.pmf(i, away_total_epv) for i in range(max_goals + 1)}
        
        # Create lookup dictionaries
        xg_pb_prob_lookup = {(sp['home_goals'], sp['away_goals']): sp['probability'] 
                           for sp in xg_game_probs_pb}
        psxg_pb_prob_lookup = {(sp['home_goals'], sp['away_goals']): sp['probability'] 
                             for sp in psxg_game_probs_pb}
        
        # Get actual scoreline
        actual_home = int(row['home_goals'])
        actual_away = int(row['away_goals'])
        
        # Store match scorelines temporarily
        match_scorelines = []
        
        # Iterate through all possible scorelines
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                
                # Get probabilities from different methods
                xg_pb_prob = xg_pb_prob_lookup.get((home_goals, away_goals), 0.0)
                psxg_pb_prob = psxg_pb_prob_lookup.get((home_goals, away_goals), 0.0)
                xg_total_poisson_prob = home_total_xg_probs[home_goals] * away_total_xg_probs[away_goals]
                epv_poisson_prob = home_epv_probs[home_goals] * away_epv_probs[away_goals]
                
                # Skip if all probabilities are negligible
                if (xg_pb_prob < 1e-10 and xg_total_poisson_prob < 1e-10 and 
                    psxg_pb_prob < 1e-10 and epv_poisson_prob < 1e-10):
                    continue
                
                # Check if this is the actual scoreline
                is_actual = (home_goals == actual_home and away_goals == actual_away)
                
                match_scorelines.append({
                    'match_id': match_id,
                    'league_id': row['league_id'],
                    'match_date': row['match_date'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'days_ago': row['days_ago'],
                    'is_actual': is_actual,
                    'xg_pb_prob_raw': xg_pb_prob,
                    'psxg_pb_prob_raw': psxg_pb_prob,
                    'xg_total_poisson_prob_raw': xg_total_poisson_prob,
                    'epv_poisson_prob_raw': epv_poisson_prob,
                })
        
        if not match_scorelines:
            continue
            
        # Filter by minimum probability threshold BEFORE normalization
        filtered_scorelines = [
            scoreline for scoreline in match_scorelines
            if (scoreline['xg_pb_prob_raw'] >= min_prob_threshold or 
                scoreline['psxg_pb_prob_raw'] >= min_prob_threshold or 
                scoreline['xg_total_poisson_prob_raw'] >= min_prob_threshold or
                scoreline['epv_poisson_prob_raw'] >= min_prob_threshold or 
                scoreline['is_actual'])
        ]
        
        # Normalize probabilities within the filtered set
        total_xg_pb_prob = sum(s['xg_pb_prob_raw'] for s in filtered_scorelines)
        total_psxg_pb_prob = sum(s['psxg_pb_prob_raw'] for s in filtered_scorelines)
        total_xg_total_poisson_prob = sum(s['xg_total_poisson_prob_raw'] for s in filtered_scorelines)
        total_epv_poisson_prob = sum(s['epv_poisson_prob_raw'] for s in filtered_scorelines)
        
        # Apply normalization and weighting
        remaining_weight = 1.0 - goals_weight
        current_match_data = []
        
        for scoreline in filtered_scorelines:
            # Normalize probabilities
            norm_pb_xg_prob = scoreline['xg_pb_prob_raw'] / total_xg_pb_prob if total_xg_pb_prob > 0 else 0
            norm_pb_psxg_prob = scoreline['psxg_pb_prob_raw'] / total_psxg_pb_prob if total_psxg_pb_prob > 0 else 0
            norm_p_xg_total_prob = scoreline['xg_total_poisson_prob_raw'] / total_xg_total_poisson_prob if total_xg_total_poisson_prob > 0 else 0
            norm_epv_prob = scoreline['epv_poisson_prob_raw'] / total_epv_poisson_prob if total_epv_poisson_prob > 0 else 0
            
            # Calculate final weight
            if scoreline['is_actual']:
                # Actual scoreline gets boost
                final_weight = goals_weight + (remaining_weight * (
                    xg_weight * norm_p_xg_total_prob + 
                    psxg_weight * norm_pb_psxg_prob +
                    epv_weight * norm_epv_prob
                ))
            else:
                # Non-actual scorelines
                final_weight = remaining_weight * (
                    xg_weight * norm_p_xg_total_prob + 
                    psxg_weight * norm_pb_psxg_prob +
                    epv_weight * norm_epv_prob
                )
            
            current_match_data.append({
                'match_id': scoreline['match_id'],
                'league_id': row['league_id'],
                'match_date': scoreline['match_date'].date(),
                'home_team': scoreline['home_team'],
                'away_team': scoreline['away_team'],
                'home_goals': scoreline['home_goals'],
                'away_goals': scoreline['away_goals'],
                'weight': final_weight,
                'days_ago': scoreline['days_ago'],
                'is_actual': scoreline['is_actual'],
                'poisson_binomial_xg_prob': norm_pb_xg_prob,
                'poisson_binomial_psxg_prob': norm_pb_psxg_prob,
                'poisson_xg_total_prob': norm_p_xg_total_prob,
                'poisson_epv_prob': norm_epv_prob
            })
        
        # Normalize weights for this match
        total_weight_for_match = sum(s['weight'] for s in current_match_data)
        if total_weight_for_match > 0:
            for s in current_match_data:
                s['weight'] = s['weight'] / total_weight_for_match
        
        # Apply time decay and red card penalty
        time_weight = np.exp(-decay_rate * row['days_ago'])
        for s in current_match_data:
            s['weight'] = s['weight'] * red_card_penalty * time_weight 
        
        expanded_data.extend(current_match_data)
    
    return pd.DataFrame(expanded_data)

def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """
    Prepare data for PyMC model by creating team mappings and indices
    
    Parameters:
    -----------
    df : DataFrame
        Processed match data
        
    Returns:
    --------
    df : DataFrame with team indices added
    team_mapping : dict mapping team names to indices  
    n_teams : number of teams
    """
    teams = sorted(df["home_team"].unique())
    n_teams = len(teams)
    team_mapping = {team: idx for idx, team in enumerate(teams)}
    
    df = df.copy()
    df['home_idx'] = pd.Categorical(df["home_team"], categories=teams).codes
    df['away_idx'] = pd.Categorical(df["away_team"], categories=teams).codes
    
    return df, team_mapping, n_teams

# Convenience function that combines everything
def load_and_process_data(db_path: str, league: Union[str, List[str]], season: str,
                         **scoreline_kwargs) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """
    Complete pipeline: load raw data, create weighted scorelines, prepare for modeling
    
    Parameters:
    -----------
    db_path : str
        Path to SQLite database
    league : str or list
        League identifier(s)
    season : str
        Season identifier (e.g., '2023-2024')
    **scoreline_kwargs : 
        Additional arguments for create_weighted_scoreline_data
        
    Returns:
    --------
    processed_df : DataFrame ready for modeling
    team_mapping : dict of team name to index mappings
    n_teams : number of teams
    """
    # Load raw data
    match_df, shot_df, red_df, epv_df = load_football_data(db_path, league, season)
    
    # Create weighted scoreline data
    weighted_df = create_weighted_scoreline_data(
        match_df, shot_df, red_df, epv_df, **scoreline_kwargs
    )
    
    # Prepare for modeling
    processed_df, team_mapping, n_teams = prepare_model_data(weighted_df)
    
    return processed_df, team_mapping, n_teams