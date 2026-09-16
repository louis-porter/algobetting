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

# WhoScored → FotMob team name normalisation (only divergent names)
WS_TO_FM_NAMES = {
    'Manchester City':  'Man City',
    'Manchester United': 'Man United',
    'Nottingham Forest': 'Nottm Forest',
}

# Garbage time thresholds: (min_margin, from_minute)
_GARBAGE_TIME_RULES = [(4, 45), (3, 57), (2, 87)]

# gross_xT (sum of positive action-value deltas per team-match) runs on a scale
# of ~2.1 avg/team-match, well above the ~1.3 avg/team-match that actual goals
# and xG sit at (2025-26 PL season) — it's a cumulative sum over ~150-300
# actions, not a shot-calibrated probability like xG, so there's no reason for
# it to land on the same scale by construction. Used uncorrected, its Poisson
# rate pulls the blended goal total up (~2.82 -> ~2.9 observed). This rescales
# it to the xG average so it acts as a peer signal instead of an outsized one;
# relative team-to-team differences are preserved (it's a single linear scale).
# Derived from: mean xG/team-match (1.320) / mean gross_xT/team-match (2.126).
XT_TO_GOALS_SCALE = 0.621


def _garbage_time_start(goals: pd.DataFrame,
                        side_col: str = 'side',
                        min_col: str = 'min') -> float:
    """
    Given a DataFrame of goals for a single match (sorted by minute),
    returns the minute at which garbage time begins, or inf if never triggered.

    side_col values must be 'home' / 'away'.
    """
    home = 0
    away = 0
    for _, g in goals.sort_values(min_col).iterrows():
        if g[side_col] == 'home':
            home += 1
        else:
            away += 1
        margin = abs(home - away)
        minute = int(g[min_col])
        for min_margin, from_min in _GARBAGE_TIME_RULES:
            if margin >= min_margin and minute >= from_min:
                return float(minute)
    return float('inf')


def load_football_data(
    db_path: str,
    league: Union[str, List[str]],
    season: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load match, shot, red card, xT, all-shots, and match-events data.

    Returns
    -------
    match_df, shot_df, red_df, xt_df, all_shots_df, events_df, xt_actions_df
        all_shots_df  : FotMob shots (incl. penalties) — used for garbage time detection
        events_df     : WhoScored match_events (goals only) — used for garbage-time detection
        xt_actions_df : WhoScored xt_actions (passes + synthesized carries) — used for competitive xT
    """
    conn = sqlite3.connect(db_path)

    if isinstance(league, str):
        leagues = [league]
    else:
        leagues = league

    league_placeholders = ','.join(['?' for _ in leagues])

    # FotMob date filter (match_date column)
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

    # WhoScored date filter (startDate column)
    events_date_clause = ""
    events_params = leagues + [season]
    if start:
        events_date_clause += " AND DATE(startDate) >= ?"
        events_params.append(str(start))
    if end:
        events_date_clause += " AND DATE(startDate) <= ?"
        events_params.append(str(end))

    # ── Match data ────────────────────────────────────────────────────────────
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

    # ── Non-penalty shots (xG model inputs) ──────────────────────────────────
    shot_df = pd.read_sql_query(f"""
        SELECT
            match_id,
            match_date,
            league_id,
            fsd.season,
            team.team_name as team,
            side,
            fsd.min,
            fsd.eventType,
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

    # ── Red cards ─────────────────────────────────────────────────────────────
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

    # ── xT (aggregated, match-level) ────────────────────────────────────────────
    # Uses gross_xT (sum of positive action deltas only), not the net xT column —
    # net sum conflates goal-relevant progression with backward/lateral possession
    # circulation that isn't really "threat" (see project_xt_model_premier_league
    # memory / the Man City discussion: net sum systematically penalises
    # high-possession teams for a stylistic trait, not chance creation). The other
    # four blend signals (goals, xG, PSxG, Bernoulli xG) are all non-negative
    # accumulations of danger, so gross_xT is the fair peer to them.
    xt_df = pd.read_sql_query(f"""
        SELECT DISTINCT
            red.match_id,
            red.match_date,
            xt.team,
            xt.gross_xT as xT,
            xt.season,
            xt.division as league_id
        FROM np_shots red
        JOIN team_id_mapping team ON team.team_id = red.teamId
        JOIN xt ON xt.team = team.team_name AND red.match_date = DATE(xt.startDate)
        WHERE
            division IN ({league_placeholders})
            AND xt.season = ?
            {date_clause}
    """, conn, params=filtered_params)

    # ── All shots incl. penalties (FotMob) — for garbage time detection ───────
    all_shots_df = pd.read_sql_query(f"""
        SELECT
            match_id,
            match_date,
            side,
            min,
            eventType
        FROM shots
        WHERE
            league_id IN ({league_placeholders})
            AND season = ?
            {date_clause}
    """, conn, params=filtered_params)

    # ── Match events (WhoScored) — goals + one context row/match, for garbage time ──
    events_df = pd.read_sql_query(f"""
        SELECT
            matchId,
            DATE(startDate) as match_date,
            homeTeam,
            awayTeam,
            h_a,
            minute,
            isGoal,
            goalOwn
        FROM match_events
        WHERE
            division IN ({league_placeholders})
            AND season = ?
            AND (isGoal = 1 OR type = 'Start')
            {events_date_clause}
    """, conn, params=events_params)

    # ── xT actions (WhoScored passes + synthesized carries) — for competitive xT ──
    xt_actions_df = pd.read_sql_query(f"""
        SELECT
            matchId,
            DATE(startDate) as match_date,
            team,
            minute,
            xT
        FROM xt_actions
        WHERE
            division IN ({league_placeholders})
            AND season = ?
            {events_date_clause}
    """, conn, params=events_params)

    conn.close()

    # days_ago for time-decay (applied to match_df, shot_df, red_df, xt_df)
    ref_date = pd.to_datetime(match_df["match_date"]).max() if not match_df.empty else pd.Timestamp.today()
    for df in [match_df, shot_df, red_df, xt_df]:
        df["days_ago"] = (ref_date - pd.to_datetime(df["match_date"])).dt.days
        df["match_date"] = pd.to_datetime(df["match_date"])

    all_shots_df["match_date"] = pd.to_datetime(all_shots_df["match_date"])
    events_df["match_date"] = pd.to_datetime(events_df["match_date"])
    xt_actions_df["match_date"] = pd.to_datetime(xt_actions_df["match_date"])

    return match_df, shot_df, red_df, xt_df, all_shots_df, events_df, xt_actions_df


def compute_competitive_xt(events_df: pd.DataFrame, xt_actions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team gross positive xT summed only over non-garbage-time actions.

    Uses gross (positive-only) xT, matching the base xT signal in xt_df — see
    the comment on that query in load_football_data for why net sum isn't used.

    Goal timing/ownership comes from WhoScored match_events (isGoal/goalOwn/h_a) —
    used only to find the garbage-time cutoff. The actual valued actions (passes
    plus synthesized carries) come from xt_actions, since that's where the xT
    grid's per-action values live (see whoscored/xt_model.py) — match_events
    itself carries no xT column.

    Own goals are attributed to the correct team before scoreline reconstruction.
    Team names are normalised to FotMob conventions via WS_TO_FM_NAMES.

    Returns
    -------
    DataFrame with columns: match_date, team, competitive_xt
    """
    records = []

    for match_id, grp in events_df.groupby('matchId'):
        home_ws = grp['homeTeam'].iloc[0]
        away_ws = grp['awayTeam'].iloc[0]
        match_date = grp['match_date'].iloc[0]

        home_fm = WS_TO_FM_NAMES.get(home_ws, home_ws)
        away_fm = WS_TO_FM_NAMES.get(away_ws, away_ws)

        # Goals: flip h_a for own goals so the scoring side is correct
        goals = grp[grp['isGoal'] == 1].copy()
        goals['side'] = goals.apply(
            lambda r: ('away' if r['h_a'] == 'h' else 'home') if r.get('goalOwn', 0) == 1
            else ('home' if r['h_a'] == 'h' else 'away'),
            axis=1
        )

        gt_start = _garbage_time_start(goals, side_col='side', min_col='minute')

        match_actions = xt_actions_df[xt_actions_df['matchId'] == match_id]
        pre_gc = match_actions[match_actions['minute'] < gt_start]
        home_xt = pre_gc[pre_gc['team'] == home_fm]['xT'].clip(lower=0).sum()
        away_xt = pre_gc[pre_gc['team'] == away_fm]['xT'].clip(lower=0).sum()

        records.append({'match_date': match_date, 'team': home_fm, 'competitive_xt': home_xt})
        records.append({'match_date': match_date, 'team': away_fm, 'competitive_xt': away_xt})

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=['match_date', 'team', 'competitive_xt']
    )


def poisson_binomial_pmf(k: int, p_values: np.ndarray) -> float:
    """
    Calculate PMF of Poisson-Binomial distribution for k successes
    given array of success probabilities p_values.

    Uses dynamic programming approach for efficiency.
    """
    n = len(p_values)
    if k > n or k < 0:
        return 0.0

    dp = np.zeros((n + 1, k + 1))
    dp[0][0] = 1.0

    for i in range(1, n + 1):
        p = p_values[i - 1]
        dp[i][0] = dp[i - 1][0] * (1 - p)
        for j in range(1, min(i, k) + 1):
            dp[i][j] = dp[i - 1][j] * (1 - p) + dp[i - 1][j - 1] * p

    return dp[n][k]


def simulate_shots_poisson_binomial(xg_values: np.ndarray, max_goals: Optional[int] = None) -> Dict[int, float]:
    """
    Simulate goal probabilities using Poisson-Binomial distribution.
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

    Shots with no possession_id are treated as independent.
    """
    side_shots = shots[shots['side'] == side].copy()
    if side_shots.empty:
        return np.array([]), 0.0

    no_poss   = side_shots[side_shots['possession_id'].isna()]
    with_poss = side_shots[side_shots['possession_id'].notna()]

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
    Calculate red card penalty multiplier based on earliest red card timing.
    """
    match_red_cards = red_cards_df[red_cards_df['match_id'] == match_id]

    if match_red_cards.empty:
        return 1.0

    earliest_red_minute = match_red_cards['time'].min()

    if earliest_red_minute > 80:   return 0.85
    elif earliest_red_minute > 70: return 0.75
    elif earliest_red_minute > 60: return 0.65
    elif earliest_red_minute > 45: return 0.50
    elif earliest_red_minute > 30: return 0.35
    elif earliest_red_minute > 15: return 0.20
    else:                          return 0.05


def compute_red_card_proportions(red_df: pd.DataFrame, match_ids, match_length: float = 90.0) -> pd.DataFrame:
    """
    Per match_id: fraction of the match each side played a man down, keyed off
    the earliest red card for that side (a second card to the same side isn't
    modelled as "two men down" — model.py's red_att_effect/red_def_effect is a
    single scalar covariate, so only the longest-duration disadvantage per side
    is used). Feeds home_red_proportion/away_red_proportion in build_and_sample_model.

    Returns one row per match_id in `match_ids`, with 0.0 for matches that have
    no red card on that side (including matches missing from red_df entirely).
    """
    base = pd.DataFrame({'match_id': pd.unique(match_ids)})

    if red_df.empty:
        return base.assign(home_red_proportion=0.0, away_red_proportion=0.0)

    earliest = red_df.groupby(['match_id', 'side'])['time'].min().reset_index()
    earliest['proportion'] = ((match_length - earliest['time']) / match_length).clip(lower=0.0, upper=1.0)

    pivot = (earliest
             .pivot(index='match_id', columns='side', values='proportion')
             .reindex(columns=['home', 'away']))
    pivot = pivot.rename(columns={'home': 'home_red_proportion', 'away': 'away_red_proportion'})

    return (base.merge(pivot, on='match_id', how='left')
                .fillna({'home_red_proportion': 0.0, 'away_red_proportion': 0.0}))


def create_weighted_scoreline_data(
    match_df: pd.DataFrame,
    shot_df: pd.DataFrame,
    red_df: pd.DataFrame,
    xt_df: pd.DataFrame,
    all_shots_df: pd.DataFrame,
    max_goals: int = 9,
    min_prob_threshold: float = 0.000,
    decay_rate: float = 0.001,
    # Base signal weights
    goals_weight: float = 0.25,
    xg_weight: float = 0.30,
    psxg_weight: float = 0.25,
    bernoulli_weight: float = 0.10,
    xt_weight: float = 0.10,
    # Garbage-time-cleaned signal weights (0 = disabled)
    gc_goals_weight: float = 0.0,
    gc_xg_weight: float = 0.0,
    gc_psxg_weight: float = 0.0,
    gc_bernoulli_weight: float = 0.0,
    gc_xt_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Build weighted scoreline distributions for each match by blending up to ten
    evidence sources — five base signals and five garbage-time-cleaned variants.

    Base signals
    ------------
    1. actual_goals  — Poisson centred on full observed scoreline
    2. xg_total      — Poisson centred on possession-aggregated xG sum
    3. psxg_pb       — Poisson-Binomial over possession-aggregated PSxG
    4. bernoulli     — Poisson-Binomial over possession-aggregated xG
    5. xt            — Poisson centred on match xT totals

    Garbage-time-cleaned variants (gc_* weights)
    --------------------------------------------
    Same as above, but shots / goals / xT after garbage time starts are excluded.
    Garbage time is determined per match from all FotMob shots (incl. penalties)
    using _GARBAGE_TIME_RULES. xT garbage time uses the pre-merged
    'competitive_xt' column on xt_df (computed from WhoScored xt_actions).

    Set gc_* weights to 0 (default) to disable entirely.

    Parameters
    ----------
    all_shots_df : DataFrame
        FotMob shots table (includes penalties) with columns:
        match_id, side, min, eventType — used to detect garbage time start minute.
    xt_df : DataFrame
        Must contain 'xT' (base) and optionally 'competitive_xt' (gc variant).
        If 'competitive_xt' is absent, gc xT falls back to full xT.
    """
    expanded_data = []

    has_gc_xt = 'competitive_xt' in xt_df.columns

    red_props = compute_red_card_proportions(red_df, match_df['match_id']).set_index('match_id')

    for _, row in match_df.iterrows():
        match_id    = row['match_id']
        match_shots = shot_df[shot_df['match_id'] == match_id]

        if match_shots.empty:
            continue

        home_red_prop = float(red_props.at[match_id, 'home_red_proportion'])
        away_red_prop = float(red_props.at[match_id, 'away_red_proportion'])

        actual_home = int(row['home_goals'])
        actual_away = int(row['away_goals'])

        # ── Garbage time detection (FotMob all-shots) ─────────────────────────
        match_all_shots = all_shots_df[all_shots_df['match_id'] == match_id]
        all_goals = match_all_shots[match_all_shots['eventType'] == 'Goal']
        gt_start = _garbage_time_start(all_goals, side_col='side', min_col='min')

        # ── gc scoreline: subtract goals scored in garbage time ────────────────
        if gt_start < float('inf'):
            gc_home_goals_lost = int(((all_goals['side'] == 'home') & (all_goals['min'] >= gt_start)).sum())
            gc_away_goals_lost = int(((all_goals['side'] == 'away') & (all_goals['min'] >= gt_start)).sum())
        else:
            gc_home_goals_lost = 0
            gc_away_goals_lost = 0
        gc_actual_home = max(0, actual_home - gc_home_goals_lost)
        gc_actual_away = max(0, actual_away - gc_away_goals_lost)

        # ── gc shots: exclude shots at or after garbage time ──────────────────
        gc_shots = match_shots[match_shots['min'] < gt_start] if gt_start < float('inf') else match_shots

        # ── Base possession-aggregated xG ─────────────────────────────────────
        home_xg_probs,   home_xg_total   = aggregate_possession_xg(match_shots, 'home', 'expectedGoals')
        away_xg_probs,   away_xg_total   = aggregate_possession_xg(match_shots, 'away', 'expectedGoals')
        home_psxg_probs, _               = aggregate_possession_xg(match_shots, 'home', 'expectedGoalsOnTarget')
        away_psxg_probs, _               = aggregate_possession_xg(match_shots, 'away', 'expectedGoalsOnTarget')

        # ── gc possession-aggregated xG ───────────────────────────────────────
        gc_home_xg_probs,   gc_home_xg_total   = aggregate_possession_xg(gc_shots, 'home', 'expectedGoals')
        gc_away_xg_probs,   gc_away_xg_total   = aggregate_possession_xg(gc_shots, 'away', 'expectedGoals')
        gc_home_psxg_probs, _                  = aggregate_possession_xg(gc_shots, 'home', 'expectedGoalsOnTarget')
        gc_away_psxg_probs, _                  = aggregate_possession_xg(gc_shots, 'away', 'expectedGoalsOnTarget')

        # ── xT lookup ─────────────────────────────────────────────────────────
        match_xt = xt_df[xt_df['match_id'] == match_id]
        home_xt  = match_xt[match_xt['team'] == row['home_team']]['xT'].values[0]        if not match_xt.empty else 0
        away_xt  = match_xt[match_xt['team'] == row['away_team']]['xT'].values[0]        if not match_xt.empty else 0
        gc_home_xt = match_xt[match_xt['team'] == row['home_team']]['competitive_xt'].values[0] \
            if (has_gc_xt and not match_xt.empty) else home_xt
        gc_away_xt = match_xt[match_xt['team'] == row['away_team']]['competitive_xt'].values[0] \
            if (has_gc_xt and not match_xt.empty) else away_xt

        # ── Distribution 1: actual goals ──────────────────────────────────────
        actual_goals_dist = {
            (h, a): poisson.pmf(h, max(actual_home, 1e-9)) * poisson.pmf(a, max(actual_away, 1e-9))
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # ── Distribution 2: xG total ──────────────────────────────────────────
        xg_total_dist = {
            (h, a): poisson.pmf(h, max(home_xg_total, 1e-9)) * poisson.pmf(a, max(away_xg_total, 1e-9))
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # ── Distribution 3: PSxG Poisson-Binomial ─────────────────────────────
        psxg_lookup = {
            (sp['home_goals'], sp['away_goals']): sp['probability']
            for sp in simulate_game_poisson_binomial(home_psxg_probs, away_psxg_probs, max_goals)
        }

        # ── Distribution 4: Bernoulli xG Poisson-Binomial ─────────────────────
        bernoulli_lookup = {
            (sp['home_goals'], sp['away_goals']): sp['probability']
            for sp in simulate_game_poisson_binomial(home_xg_probs, away_xg_probs, max_goals)
        }

        # ── Distribution 5: xT ────────────────────────────────────────────────
        xt_dist = {
            (h, a): poisson.pmf(h, max(home_xt, 1e-9)) * poisson.pmf(a, max(away_xt, 1e-9))
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # ── gc Distribution 1: competitive actual goals ────────────────────────
        gc_actual_goals_dist = {
            (h, a): poisson.pmf(h, max(gc_actual_home, 1e-9)) * poisson.pmf(a, max(gc_actual_away, 1e-9))
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # ── gc Distribution 2: competitive xG total ───────────────────────────
        gc_xg_total_dist = {
            (h, a): poisson.pmf(h, max(gc_home_xg_total, 1e-9)) * poisson.pmf(a, max(gc_away_xg_total, 1e-9))
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # ── gc Distribution 3: competitive PSxG ───────────────────────────────
        gc_psxg_lookup = {
            (sp['home_goals'], sp['away_goals']): sp['probability']
            for sp in simulate_game_poisson_binomial(gc_home_psxg_probs, gc_away_psxg_probs, max_goals)
        }

        # ── gc Distribution 4: competitive Bernoulli xG ───────────────────────
        gc_bernoulli_lookup = {
            (sp['home_goals'], sp['away_goals']): sp['probability']
            for sp in simulate_game_poisson_binomial(gc_home_xg_probs, gc_away_xg_probs, max_goals)
        }

        # ── gc Distribution 5: competitive xT ─────────────────────────────────
        gc_xt_dist = {
            (h, a): poisson.pmf(h, max(gc_home_xt, 1e-9)) * poisson.pmf(a, max(gc_away_xt, 1e-9))
            for h, a in product(range(max_goals + 1), range(max_goals + 1))
        }

        # ── Blend all signals ─────────────────────────────────────────────────
        match_scorelines = []
        for home_goals, away_goals in product(range(max_goals + 1), range(max_goals + 1)):
            key = (home_goals, away_goals)

            final_weight = (
                goals_weight      * actual_goals_dist.get(key, 0.0) +
                xg_weight         * xg_total_dist.get(key, 0.0)     +
                psxg_weight       * psxg_lookup.get(key, 0.0)        +
                bernoulli_weight  * bernoulli_lookup.get(key, 0.0)   +
                xt_weight         * xt_dist.get(key, 0.0)            +
                gc_goals_weight   * gc_actual_goals_dist.get(key, 0.0) +
                gc_xg_weight      * gc_xg_total_dist.get(key, 0.0)    +
                gc_psxg_weight    * gc_psxg_lookup.get(key, 0.0)       +
                gc_bernoulli_weight * gc_bernoulli_lookup.get(key, 0.0) +
                gc_xt_weight      * gc_xt_dist.get(key, 0.0)
            )

            if final_weight < min_prob_threshold and key != (actual_home, actual_away):
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
                'is_actual':  key == (actual_home, actual_away),
                'home_red_proportion': home_red_prop,
                'away_red_proportion': away_red_prop,
            })

        total_m_weight = sum(s['weight'] for s in match_scorelines)
        time_decay     = np.exp(-decay_rate * row['days_ago'])

        for s in match_scorelines:
            s['weight'] = (s['weight'] / total_m_weight) * time_decay
            expanded_data.append(s)

    return pd.DataFrame(expanded_data)


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """
    Prepare data for PyMC model by creating team mappings and indices.
    """
    teams       = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    n_teams     = len(teams)
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

    Forwards all **scoreline_kwargs to create_weighted_scoreline_data, including
    gc_* weight parameters for garbage-time-cleaned signals (default 0 = disabled).
    """
    match_df, shot_df, red_df, xt_df, all_shots_df, events_df, xt_actions_df = load_football_data(
        db_path, league, season, start=start, end=end
    )

    # Rescale gross xT onto the same goals-equivalent scale as xG (see
    # XT_TO_GOALS_SCALE above) before it's used as a Poisson rate.
    xt_df['xT'] = xt_df['xT'] * XT_TO_GOALS_SCALE

    # Compute competitive xT from WhoScored xt_actions and merge into xt_df
    if not events_df.empty and not xt_actions_df.empty:
        competitive_xt_df = compute_competitive_xt(events_df, xt_actions_df)
        if not competitive_xt_df.empty:
            competitive_xt_df['competitive_xt'] = competitive_xt_df['competitive_xt'] * XT_TO_GOALS_SCALE
            xt_df = xt_df.merge(
                competitive_xt_df,
                on=['match_date', 'team'],
                how='left',
            )
            # Fallback: use full xT where no WhoScored data available
            xt_df['competitive_xt'] = xt_df['competitive_xt'].fillna(xt_df['xT'])

    weighted_df  = create_weighted_scoreline_data(
        match_df, shot_df, red_df, xt_df, all_shots_df, **scoreline_kwargs
    )

    processed_df, team_mapping, n_teams = prepare_model_data(weighted_df)

    return processed_df, team_mapping, n_teams
