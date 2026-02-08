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
        FROM np_matches fmd
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
        FROM np_shots fsd
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
        FROM np_shots red
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
                                 goals_weight: float = 0.25, # The weight for 'Actual Performance'
                                 xg_weight: float = 0.40,
                                 psxg_weight: float = 0.25,
                                 epv_weight: float = 0.1) -> pd.DataFrame:
    
    expanded_data = []
    
    for idx, row in match_df.iterrows():
        match_id = row['match_id']
        match_shots = shot_df[shot_df["match_id"] == match_id]
        
        if match_shots.empty:
            continue
            
        # 1. --- EXTRACT DATA ---
        actual_home = int(row['home_goals'])
        actual_away = int(row['away_goals'])
        
        home_xg_total = match_shots[match_shots['side'] == 'home']['expectedGoals'].sum()
        away_xg_total = match_shots[match_shots['side'] == 'away']['expectedGoals'].sum()

        home_psxg_shots = match_shots[match_shots['side'] == 'home']['expectedGoalsOnTarget'].values
        away_psxg_shots = match_shots[match_shots['side'] == 'away']['expectedGoalsOnTarget'].values

        match_epv = epv_df[epv_df["match_id"] == match_id]
        home_total_epv = match_epv[match_epv['team'] == row['home_team']]['EPV'].values[0] if not match_epv.empty else 0
        away_total_epv = match_epv[match_epv['team'] == row['away_team']]['EPV'].values[0] if not match_epv.empty else 0

        # 2. --- GENERATE POISSON DISTRIBUTIONS FOR EVERY EXPERT ---
        # This is the "Bayesian Fix": converting the actual score into a Poisson parameter
        actual_goals_dist = {
            (h, a): poisson.pmf(h, actual_home) * poisson.pmf(a, actual_away)
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }
        
        xg_total_dist = {
            (h, a): poisson.pmf(h, home_xg_total) * poisson.pmf(a, away_xg_total)
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # PB (Poisson-Binomial) for PSxG (Individual shot quality)
        psxg_game_probs = simulate_game_poisson_binomial(
            np.nan_to_num(home_psxg_shots, 0.0), 
            np.nan_to_num(away_psxg_shots, 0.0), 
            max_goals
        )
        psxg_pb_lookup = {(sp['home_goals'], sp['away_goals']): sp['probability'] for sp in psxg_game_probs}

        epv_dist = {
            (h, a): poisson.pmf(h, home_total_epv) * poisson.pmf(a, away_total_epv)
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # 3. --- BLEND AND NORMALIZE ---
        match_scorelines = []
        for home_goals, away_goals in product(range(max_goals + 1), range(max_goals + 1)):
            
            # Retrieve probabilities
            p_actual = actual_goals_dist.get((home_goals, away_goals), 0.0)
            p_xg = xg_total_dist.get((home_goals, away_goals), 0.0)
            p_psxg = psxg_pb_lookup.get((home_goals, away_goals), 0.0)
            p_epv = epv_dist.get((home_goals, away_goals), 0.0)

            # Weight the experts
            # No hard-coding! We sum the weighted probabilities.
            final_weight = (
                (goals_weight * p_actual) + 
                (xg_weight * p_xg) + 
                (psxg_weight * p_psxg) + 
                (epv_weight * p_epv)
            )

            if final_weight < min_prob_threshold and not (home_goals == actual_home and away_goals == actual_away):
                continue

            match_scorelines.append({
                'match_id': match_id,
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_goals': home_goals,
                'away_goals': away_goals,
                'weight': final_weight,
                'days_ago': row['days_ago'],
                'is_actual': (home_goals == actual_home and away_goals == actual_away)
            })

        # Final match-level weight normalization
        total_m_weight = sum(s['weight'] for s in match_scorelines)
        time_decay = np.exp(-decay_rate * row['days_ago'])
        
        for s in match_scorelines:
            # Normalize so the match total is 1.0, then apply time decay
            s['weight'] = (s['weight'] / total_m_weight) * time_decay
            expanded_data.append(s)

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