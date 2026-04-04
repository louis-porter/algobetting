"""
team_stats.py — standard per-team per-season metric aggregation

Single source of truth for the attack/defence metric definitions used
across spider and radar charts.
"""

import sqlite3
import pandas as pd

# ── Standard metric definitions ───────────────────────────────────────────────
# (axis label, DataFrame column, invert, fmt)
# invert=True → lower raw value is better on the chart

ATTACK_METRICS = [
    ('npxG p/g',         'npxG_for',            False, '{:.2f}'),
    ('npxG / shot',      'npxG_per_shot_for',    False, '{:.2f}'),
    ('Shots p/g',        'shots_for',             False, '{:.1f}'),
    ('Set piece xG',     'sp_xg_for',             False, '{:.2f}'),
    ('Box touches p/g',  'box_touches_for',       False, '{:.1f}'),
    ('Possession %',     'possession',            False, '{:.1f}'),
]

DEFENCE_METRICS = [
    ('npxGA p/g',        'npxGA',                True,  '{:.2f}'),
    ('npxGA / shot',     'npxG_per_shot_ag',     True,  '{:.2f}'),
    ('Shots conceded',   'shots_against',         True,  '{:.1f}'),
    ('Set piece xGA',    'sp_xg_against',         True,  '{:.2f}'),
    ('Opp box touches',  'box_touches_against',   True,  '{:.1f}'),
    ('Opp pass %',       'opp_pass_pct',          True,  '{:.1f}'),
]


def get_standard_stats(db_path, league, seasons=None):
    """
    Return per-team per-season averages for all standard metrics.

    Parameters
    ----------
    db_path : str       Path to fotmob.db
    league  : str       e.g. 'Superligaen'
    seasons : list[str] Optional filter, e.g. ['2024-2025', '2025-2026']

    Returns
    -------
    pd.DataFrame  columns: team_name, season, + all metric columns
    """
    conn = sqlite3.connect(db_path)

    # ── 1. npxG per team per match ────────────────────────────────────────────
    lshots = pd.read_sql(f"""
        SELECT match_id, season, teamId AS team_id,
               SUM(expectedGoals) AS npxg_for,
               COUNT(*)           AS np_shots_for
        FROM np_shots
        WHERE league_id = '{league}'
        GROUP BY match_id, season, teamId
    """, conn)

    # Opponent npxG via self-join
    opp = (lshots[['match_id', 'team_id', 'npxg_for', 'np_shots_for']]
           .rename(columns={'team_id': 'opp_id',
                            'npxg_for': 'npxga',
                            'np_shots_for': 'np_shots_ag'}))
    lshots = (lshots
              .merge(opp, on='match_id', how='left')
              .query('team_id != opp_id')
              .drop(columns='opp_id'))

    # Team names
    names = pd.read_sql("SELECT team_id, team_name FROM team_id_mapping", conn)
    lshots = lshots.merge(names, on='team_id', how='left')

    # ── 2. Match stats — unpivot home/away ────────────────────────────────────
    ms = pd.read_sql(f"""
        SELECT match_id, season, home_team, away_team,
               home_total_shots,               away_total_shots,
               home_expected_goals_set_play,   away_expected_goals_set_play,
               home_touches_opp_box,           away_touches_opp_box,
               home_BallPossesion,             away_BallPossesion,
               home_accurate_passes_pct,       away_accurate_passes_pct
        FROM match_stats
        WHERE league_id = '{league}'
    """, conn)
    conn.close()

    num_cols = [
        'home_total_shots', 'away_total_shots',
        'home_expected_goals_set_play', 'away_expected_goals_set_play',
        'home_touches_opp_box', 'away_touches_opp_box',
        'home_BallPossesion', 'away_BallPossesion',
        'home_accurate_passes_pct', 'away_accurate_passes_pct',
    ]
    for c in num_cols:
        ms[c] = pd.to_numeric(ms[c], errors='coerce')

    def _perspective(ms, team_col, for_prefix, ag_prefix):
        return ms[['match_id', team_col,
                   f'{for_prefix}_total_shots',
                   f'{for_prefix}_expected_goals_set_play',
                   f'{for_prefix}_touches_opp_box',
                   f'{for_prefix}_BallPossesion',
                   f'{for_prefix}_accurate_passes_pct',
                   f'{ag_prefix}_total_shots',
                   f'{ag_prefix}_expected_goals_set_play',
                   f'{ag_prefix}_touches_opp_box',
                   f'{ag_prefix}_accurate_passes_pct',
                   ]].rename(columns={
            team_col:                              'team_id',
            f'{for_prefix}_total_shots':           'shots_for',
            f'{for_prefix}_expected_goals_set_play':'sp_xg_for',
            f'{for_prefix}_touches_opp_box':       'box_touches_for',
            f'{for_prefix}_BallPossesion':         'possession',
            f'{for_prefix}_accurate_passes_pct':   'pass_pct_for',
            f'{ag_prefix}_total_shots':            'shots_against',
            f'{ag_prefix}_expected_goals_set_play':'sp_xg_against',
            f'{ag_prefix}_touches_opp_box':        'box_touches_against',
            f'{ag_prefix}_accurate_passes_pct':    'opp_pass_pct',
        })

    ms_long = pd.concat([
        _perspective(ms, 'home_team', 'home', 'away'),
        _perspective(ms, 'away_team', 'away', 'home'),
    ])

    # ── 3. Merge + derive ─────────────────────────────────────────────────────
    df = lshots.merge(ms_long, on=['match_id', 'team_id'], how='left')
    df['npxG_per_shot_for'] = df['npxg_for'] / df['np_shots_for']
    df['npxG_per_shot_ag']  = df['npxga']    / df['np_shots_ag']

    if seasons:
        df = df[df['season'].isin(seasons)]

    # ── 4. Season averages ────────────────────────────────────────────────────
    stats = df.groupby(['team_name', 'season']).agg(
        npxG_for            = ('npxg_for',           'mean'),
        npxGA               = ('npxga',              'mean'),
        npxG_per_shot_for   = ('npxG_per_shot_for',  'mean'),
        npxG_per_shot_ag    = ('npxG_per_shot_ag',   'mean'),
        shots_for           = ('shots_for',           'mean'),
        shots_against       = ('shots_against',       'mean'),
        sp_xg_for           = ('sp_xg_for',           'mean'),
        sp_xg_against       = ('sp_xg_against',       'mean'),
        box_touches_for     = ('box_touches_for',     'mean'),
        box_touches_against = ('box_touches_against', 'mean'),
        possession          = ('possession',          'mean'),
        opp_pass_pct        = ('opp_pass_pct',        'mean'),
    ).round(3).reset_index()

    return stats
