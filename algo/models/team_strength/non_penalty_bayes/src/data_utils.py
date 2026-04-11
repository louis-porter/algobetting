# =============================================================================
# src/data_utils.py
# =============================================================================
import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import poisson
from itertools import product
from typing import Dict, List, Tuple, Optional, Union
from datetime import date

def load_football_data(
    db_path: str,
    league: Union[str, List[str]],
    season: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load match, shot, red card, and EPV data from SQLite database.

    Parameters:
    -----------
    db_path : str
        Path to the SQLite database
    league : str or list
        League identifier(s) (e.g., 'Premier_League')
    season : str
        Season identifier (e.g., '2023-2024')
    start : date, optional
        Earliest match date to include (inclusive)
    end : date, optional
        Latest match date to include (inclusive)

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

    # Build optional date filter clauses and params
    date_clause = ""
    date_params: list = []
    if start:
        date_clause += " AND match_date >= ?"
        date_params.append(str(start))
    if end:
        date_clause += " AND match_date <= ?"
        date_params.append(str(end))

    base_params = leagues + [season]
    filtered_params = base_params + date_params

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
            {date_clause}
    """, conn, params=filtered_params)

    # Load shot data — includes possession_id joined from shot_possession_id table
    shot_df = pd.read_sql_query(f"""
        SELECT 
            match_id,
            match_date,
            league_id,
            fsd.season,
            team.team_name as team,
            side,
            expectedGoals,
            expectedGoalsOnTarget,
            possession_id
        FROM np_shots fsd
            JOIN team_id_mapping team ON team.team_id = fsd.teamId
        LEFT JOIN shot_possession_id spi
            ON matchDate = match_date AND fsd.shot_rank = spi.shot_rank AND fsd.team_name = spi.teamName
        WHERE
            league_id IN ({league_placeholders})
            AND fsd.season = ?
            {date_clause}
    """, conn, params=filtered_params)

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
            {date_clause}
    """, conn, params=filtered_params)

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
            {date_clause}
    """, conn, params=filtered_params)

    conn.close()

    # Add days_ago calculation for all dataframes
    for df in [match_df, shot_df, red_df, epv_df]:
        df["days_ago"] = (pd.to_datetime(df["match_date"]).max() - pd.to_datetime(df["match_date"])).dt.days
        df["match_date"] = pd.to_datetime(df["match_date"])

    return match_df, shot_df, red_df, epv_df


def poisson_binomial_pmf(k: int, p_values: np.ndarray) -> float:
    """
    Calculate PMF of Poisson-Binomial distribution for k successes
    given array of success probabilities p_values.

    Uses dynamic programming approach for efficiency.
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
            dp[i][j] = dp[i - 1][j] * (1 - p) + dp[i - 1][j - 1] * p

    return dp[n][k]


def simulate_shots_poisson_binomial(xg_values: np.ndarray, max_goals: Optional[int] = None) -> Dict[int, float]:
    """
    Simulate goal probabilities using Poisson-Binomial distribution.

    Args:
        xg_values: array of xG values for individual shots / possessions
        max_goals: maximum number of goals to consider (default: number of trials)

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
        if prob > 1e-10:
            goal_probs[goals] = prob

    return goal_probs


def simulate_game_poisson_binomial(home_xg_shots: np.ndarray, away_xg_shots: np.ndarray,
                                   max_goals: int = 9) -> List[Dict]:
    """
    Simulate a game using Poisson-Binomial distribution for each team.

    Args:
        home_xg_shots: array of xG values for home team (one per possession)
        away_xg_shots: array of xG values for away team (one per possession)
        max_goals: maximum goals to consider for each team

    Returns:
        list of dicts with home_goals, away_goals, and probability
    """
    home_probs = simulate_shots_poisson_binomial(home_xg_shots, max_goals)
    away_probs = simulate_shots_poisson_binomial(away_xg_shots, max_goals)

    game_probs = []
    for (h_goals, h_prob), (a_goals, a_prob) in product(home_probs.items(), away_probs.items()):
        combined_prob = h_prob * a_prob
        if combined_prob > 1e-6:
            game_probs.append({
                'home_goals': h_goals,
                'away_goals': a_goals,
                'probability': combined_prob
            })

    return game_probs


def aggregate_possession_xg(shots: pd.DataFrame, side: str, col: str) -> Tuple[np.ndarray, float]:
    """
    Collapse correlated rebound shots into one xG value per possession using
    the complement rule: P(any shot scores) = 1 - prod(1 - xg_i).

    Shots with no possession_id (NULL) are treated as independent and passed
    through unchanged.

    Parameters:
    -----------
    shots : DataFrame
        Shot-level data for a single match, must contain columns:
        'side', 'possession_id', and the xG column specified by `col`
    side : str
        'home' or 'away'
    col : str
        Column name to aggregate, e.g. 'expectedGoals' or 'expectedGoalsOnTarget'

    Returns:
    --------
    possession_probs : np.ndarray
        One probability value per possession (or independent shot).
        This is the array to pass into simulate_shots_poisson_binomial.
    xg_total : float
        Sum of possession_probs — use as the lambda for a Poisson distribution.
    """
    side_shots = shots[shots['side'] == side].copy()
    if side_shots.empty:
        return np.array([]), 0.0

    # Shots without a possession_id are independent — pass through as-is
    no_poss = side_shots[side_shots['possession_id'].isna()]
    with_poss = side_shots[side_shots['possession_id'].notna()]

    # Collapse each possession: P(score) = 1 - prod(1 - xg_i)
    possession_xg = (
        with_poss
        .groupby('possession_id')[col]
        .apply(lambda xgs: 1 - np.prod(1 - np.nan_to_num(xgs.values, nan=0.0)))
    ).values

    independent_xg = np.nan_to_num(no_poss[col].values, nan=0.0)
    all_probs = np.concatenate([possession_xg, independent_xg])

    return all_probs, float(all_probs.sum())


def calculate_red_card_penalty(red_cards_df: pd.DataFrame, match_id: str) -> float:
    """
    Calculate red card penalty for a specific match based on earliest red card timing.

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
        return 1.0

    earliest_red_minute = match_red_cards['time'].min()

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
    else:
        return 0.05


def create_weighted_scoreline_data(match_df: pd.DataFrame,
                                   shot_df: pd.DataFrame,
                                   red_df: pd.DataFrame,
                                   epv_df: pd.DataFrame,
                                   max_goals: int = 9,
                                   min_prob_threshold: float = 0.001,
                                   decay_rate: float = 0.001,
                                   goals_weight: float = 0.25,
                                   xg_weight: float = 0.30,
                                   psxg_weight: float = 0.25,
                                   bernoulli_weight: float = 0.10,
                                   epv_weight: float = 0.10) -> pd.DataFrame:
    """
    Build weighted scoreline distributions for each match by blending five
    evidence sources:

        1. actual_goals  — Poisson centred on observed scoreline
        2. xg_total      — Poisson centred on possession-aggregated xG sum
        3. psxg_pb       — Poisson-Binomial over possession-aggregated PSxG array
        4. bernoulli     — Poisson-Binomial over possession-aggregated xG array
                           (same array as xg_total but modelled as discrete trials)
        5. epv           — Poisson centred on match EPV totals

    Shots that share a possession_id are treated as dependent (rebounds) by
    collapsing them to a single possession probability via the complement rule
    before any distribution is computed. Shots with NULL possession_id are
    treated as independent.

    Parameters:
    -----------
    goals_weight : float
        Weight for the actual observed goals distribution
    xg_weight : float
        Weight for the Poisson xG total distribution
    psxg_weight : float
        Weight for the Poisson-Binomial PSxG distribution
    bernoulli_weight : float
        Weight for the Poisson-Binomial xG possession distribution
    epv_weight : float
        Weight for the EPV distribution

    Note: weights do not need to sum to 1 — they are used as relative scaling
    factors within the blend and the result is normalised per match.
    """
    expanded_data = []

    for idx, row in match_df.iterrows():
        match_id = row['match_id']
        match_shots = shot_df[shot_df["match_id"] == match_id]

        if match_shots.empty:
            continue

        actual_home = int(row['home_goals'])
        actual_away = int(row['away_goals'])

        # ------------------------------------------------------------------
        # Possession-aggregated xG arrays and totals
        # ------------------------------------------------------------------
        home_xg_probs, home_xg_total = aggregate_possession_xg(match_shots, 'home', 'expectedGoals')
        away_xg_probs, away_xg_total = aggregate_possession_xg(match_shots, 'away', 'expectedGoals')

        home_psxg_probs, _ = aggregate_possession_xg(match_shots, 'home', 'expectedGoalsOnTarget')
        away_psxg_probs, _ = aggregate_possession_xg(match_shots, 'away', 'expectedGoalsOnTarget')

        # ------------------------------------------------------------------
        # EPV (match-level, no shot decomposition available)
        # ------------------------------------------------------------------
        match_epv = epv_df[epv_df["match_id"] == match_id]
        home_total_epv = match_epv[match_epv['team'] == row['home_team']]['EPV'].values[0] if not match_epv.empty else 0
        away_total_epv = match_epv[match_epv['team'] == row['away_team']]['EPV'].values[0] if not match_epv.empty else 0

        # ------------------------------------------------------------------
        # Distribution 1: actual goals — Poisson on observed scoreline
        # ------------------------------------------------------------------
        actual_goals_dist = {
            (h, a): poisson.pmf(h, max(actual_home, 1e-9)) * poisson.pmf(a, max(actual_away, 1e-9))
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # ------------------------------------------------------------------
        # Distribution 2: xG total — Poisson on possession-aggregated xG sum
        # ------------------------------------------------------------------
        xg_total_dist = {
            (h, a): poisson.pmf(h, max(home_xg_total, 1e-9)) * poisson.pmf(a, max(away_xg_total, 1e-9))
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # ------------------------------------------------------------------
        # Distribution 3: PSxG — Poisson-Binomial on possession PSxG array
        # ------------------------------------------------------------------
        psxg_game_probs = simulate_game_poisson_binomial(home_psxg_probs, away_psxg_probs, max_goals)
        psxg_pb_lookup = {(sp['home_goals'], sp['away_goals']): sp['probability'] for sp in psxg_game_probs}

        # ------------------------------------------------------------------
        # Distribution 4: Bernoulli xG — Poisson-Binomial on possession xG array
        # ------------------------------------------------------------------
        bernoulli_game_probs = simulate_game_poisson_binomial(home_xg_probs, away_xg_probs, max_goals)
        bernoulli_lookup = {(sp['home_goals'], sp['away_goals']): sp['probability'] for sp in bernoulli_game_probs}

        # ------------------------------------------------------------------
        # Distribution 5: EPV — Poisson on EPV totals
        # ------------------------------------------------------------------
        epv_dist = {
            (h, a): poisson.pmf(h, max(home_total_epv, 1e-9)) * poisson.pmf(a, max(away_total_epv, 1e-9))
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # ------------------------------------------------------------------
        # Blend all five distributions
        # ------------------------------------------------------------------
        match_scorelines = []
        for home_goals, away_goals in product(range(max_goals + 1), range(max_goals + 1)):

            p_actual   = actual_goals_dist.get((home_goals, away_goals), 0.0)
            p_xg       = xg_total_dist.get((home_goals, away_goals), 0.0)
            p_psxg     = psxg_pb_lookup.get((home_goals, away_goals), 0.0)
            p_bernoulli = bernoulli_lookup.get((home_goals, away_goals), 0.0)
            p_epv      = epv_dist.get((home_goals, away_goals), 0.0)

            final_weight = (
                (goals_weight    * p_actual)    +
                (xg_weight       * p_xg)        +
                (psxg_weight     * p_psxg)      +
                (bernoulli_weight * p_bernoulli) +
                (epv_weight      * p_epv)
            )

            if final_weight < min_prob_threshold and not (home_goals == actual_home and away_goals == actual_away):
                continue

            match_scorelines.append({
                'match_id':   match_id,
                'match_date': row['match_date'],
                'home_team':  row['home_team'],
                'away_team':  row['away_team'],
                'home_goals': home_goals,
                'away_goals': away_goals,
                'weight':     final_weight,
                'days_ago':   row['days_ago'],
                'is_actual':  (home_goals == actual_home and away_goals == actual_away)
            })

        total_m_weight = sum(s['weight'] for s in match_scorelines)
        time_decay = np.exp(-decay_rate * row['days_ago'])

        for s in match_scorelines:
            s['weight'] = (s['weight'] / total_m_weight) * time_decay
            expanded_data.append(s)

    return pd.DataFrame(expanded_data)


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """
    Prepare data for PyMC model by creating team mappings and indices.

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


def load_and_process_data(
    db_path: str,
    league: Union[str, List[str]],
    season: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    **scoreline_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """
    Complete pipeline: load raw data, create weighted scorelines, prepare for modelling.

    Parameters:
    -----------
    db_path : str
        Path to SQLite database
    league : str or list
        League identifier(s)
    season : str
        Season identifier (e.g., '2023-2024')
    start : date, optional
        Earliest match date to include (inclusive)
    end : date, optional
        Latest match date to include (inclusive)
    **scoreline_kwargs :
        Additional arguments forwarded to create_weighted_scoreline_data
        (e.g. goals_weight, xg_weight, psxg_weight, bernoulli_weight, epv_weight)

    Returns:
    --------
    processed_df : DataFrame ready for modelling
    team_mapping : dict of team name to index mappings
    n_teams : number of teams
    """
    match_df, shot_df, red_df, epv_df = load_football_data(
        db_path, league, season, start=start, end=end
    )

    weighted_df = create_weighted_scoreline_data(
        match_df, shot_df, red_df, epv_df, **scoreline_kwargs
    )

    processed_df, team_mapping, n_teams = prepare_model_data(weighted_df)

    return processed_df, team_mapping, n_teams