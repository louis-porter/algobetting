"""
Expected Threat (xT) — fit from our own match_events data.

Follows Karun Singh's methodology (https://karun.in/blog/expected-threat.html):
pitch is discretized into a grid, each zone gets a probability of shooting vs.
moving on, a goal probability if shooting, and a transition matrix if moving,
then the value surface is solved by iterating:

    xT[z] = shoot_prob[z] * goal_prob[z] + move_prob[z] * sum_over(z') T[z->z'] * xT[z']

An action's value is the xT delta between its end zone and its start zone.

Unlike the existing `EPV` column (which looks up a static, pre-fit grid
borrowed from someone else's model — see EPV_grid.csv / add_epv_to_events.py),
this grid is fit directly from the Premier League match_events already
collected in this database, and gets refit every time it's rerun as more
matches accumulate.

Coordinate note: WhoScored's x/y in this database are already oriented so
that x=100 is the goal the team-in-possession is attacking, for every team
in every period (verified empirically — shot x averages ~85 for every
team/period/half combination, not just for whichever side happens to be
kicking left-to-right). So no attacking-direction normalization is needed
here, unlike a naive read of the raw Opta convention would suggest.

WhoScored/Opta events don't carry an explicit "Carry" action with an end
location the way passes do (dribbles only have a start touch). So carries
are synthesized: whenever the same player produces two consecutive events
in the raw stream (nothing else happened in between), the gap between the
first event's location and the second event's location is treated as a
carry. This is the standard heuristic used by public SPADL/xT implementations.

Writes two derived tables (full replace each run — cheap to recompute from
match_events, so no incremental dedup bookkeeping):
  - xt_actions : every valued pass/carry action, matchId/team/player level
  - xt         : matchId/team aggregate, same shape as the existing `epv` table

Also writes the fitted grid + fit metadata next to this file for inspection:
  - xt_grid.csv
  - xt_model_meta.json
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = r"/Users/admin/dev/algobetting/infra/data/db/fotmob.db"
_HERE = Path(__file__).parent
GRID_CSV_PATH = _HERE / "xt_grid.csv"
META_JSON_PATH = _HERE / "xt_model_meta.json"

N_ROWS = 12   # y bins (Karun Singh's blog uses a 16x12 grid)
N_COLS = 16   # x bins
N_ITERATIONS = 5

TEAM_NAME_MAPPING = {
    'Nottingham Forest': 'Nottm Forest',
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
}

# Events that aren't on-the-ball actions — never treated as one end of a carry.
NON_ACTION_TYPES = {
    'SubstitutionOff', 'SubstitutionOn', 'FormationChange', 'FormationSet',
    'Start', 'End', 'Card', 'OffsideGiven', 'OffsideProvoked', 'OffsidePass',
}

MAX_CARRY_SECONDS = 15.0
MIN_CARRY_DIST = 3.0    # in WhoScored 0-100 units — filters GPS/event-timing noise
MAX_CARRY_DIST = 60.0   # filters mis-linked/glitched consecutive events


# ─────────────────────────────────────────────
# Load + prep
# ─────────────────────────────────────────────

def load_events(db_name: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_name)
    try:
        df = pd.read_sql_query(
            """
            SELECT matchId, startDate, period, minute, second, teamId, playerId,
                   homeTeam, awayTeam, h_a, type, outcomeType, isShot, isGoal,
                   x, y, endX, endY, season, division
            FROM match_events
            WHERE period IN ('FirstHalf', 'SecondHalf')
            """,
            conn,
        )
    finally:
        conn.close()

    df['minute'] = pd.to_numeric(df['minute'], errors='coerce').fillna(0)
    df['second'] = pd.to_numeric(df['second'], errors='coerce').fillna(0)
    df['team'] = df['homeTeam'].where(df['h_a'] == 'h', df['awayTeam']).replace(TEAM_NAME_MAPPING)
    df['opponent'] = df['awayTeam'].where(df['h_a'] == 'h', df['homeTeam']).replace(TEAM_NAME_MAPPING)
    return df


def _time_seconds(df: pd.DataFrame) -> pd.Series:
    period_offset = df['period'].map({'FirstHalf': 0, 'SecondHalf': 45 * 60}).fillna(0)
    return period_offset + df['minute'] * 60 + df['second']


def synthesize_carries(df: pd.DataFrame) -> pd.DataFrame:
    """Derive implicit carry actions from consecutive same-player touches."""
    df = df.sort_values(['matchId', 'period', 'minute', 'second']).reset_index(drop=True)
    df = df.assign(_t=_time_seconds(df))

    nxt = df.shift(-1)
    same_chain = (
        (df['matchId'] == nxt['matchId'])
        & (df['period'] == nxt['period'])
        & (df['teamId'] == nxt['teamId'])
        & (df['playerId'] == nxt['playerId'])
        & (~df['type'].isin(NON_ACTION_TYPES))
        & (df['isShot'] != 1)
        & df['x'].notna() & df['y'].notna()
        & nxt['x'].notna() & nxt['y'].notna()
    )

    dt = nxt['_t'] - df['_t']
    dist = np.hypot(nxt['x'] - df['x'], nxt['y'] - df['y'])

    mask = (
        same_chain
        & (dt >= 0) & (dt <= MAX_CARRY_SECONDS)
        & (dist >= MIN_CARRY_DIST) & (dist <= MAX_CARRY_DIST)
    )

    carries = pd.DataFrame({
        'matchId': df.loc[mask, 'matchId'].values,
        'startDate': df.loc[mask, 'startDate'].values,
        'period': df.loc[mask, 'period'].values,
        'minute': df.loc[mask, 'minute'].values,
        'second': df.loc[mask, 'second'].values,
        'teamId': df.loc[mask, 'teamId'].values,
        'playerId': df.loc[mask, 'playerId'].values,
        'team': df.loc[mask, 'team'].values,
        'opponent': df.loc[mask, 'opponent'].values,
        'season': df.loc[mask, 'season'].values,
        'division': df.loc[mask, 'division'].values,
        'x': df.loc[mask, 'x'].values,
        'y': df.loc[mask, 'y'].values,
        'endX': nxt.loc[mask, 'x'].values,
        'endY': nxt.loc[mask, 'y'].values,
        'action_type': 'carry',
    })
    return carries


def build_moves_and_shots(df: pd.DataFrame):
    keep_cols = ['matchId', 'startDate', 'period', 'minute', 'second', 'teamId',
                 'playerId', 'team', 'opponent', 'season', 'division', 'x', 'y', 'endX', 'endY']

    passes = df[
        (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')
        & df[['x', 'y', 'endX', 'endY']].notna().all(axis=1)
    ][keep_cols].copy()
    passes['action_type'] = 'pass'

    carries = synthesize_carries(df)

    moves = pd.concat([passes, carries], ignore_index=True)

    shots = df[(df['isShot'] == 1) & df['x'].notna() & df['y'].notna()][
        ['matchId', 'teamId', 'playerId', 'x', 'y', 'isGoal']
    ].copy()

    return moves, shots


# ─────────────────────────────────────────────
# Grid fit
# ─────────────────────────────────────────────

def zone_indices(x, y, n_cols=N_COLS, n_rows=N_ROWS):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    col = np.clip((x / 100.0 * n_cols).astype(int), 0, n_cols - 1)
    row = np.clip((y / 100.0 * n_rows).astype(int), 0, n_rows - 1)
    return row, col


def fit_xt_grid(moves: pd.DataFrame, shots: pd.DataFrame,
                 n_rows: int = N_ROWS, n_cols: int = N_COLS,
                 n_iterations: int = N_ITERATIONS):
    moves = moves.copy()
    shots = shots.copy()

    moves['row'], moves['col'] = zone_indices(moves['x'].values, moves['y'].values, n_cols, n_rows)
    moves['row2'], moves['col2'] = zone_indices(moves['endX'].values, moves['endY'].values, n_cols, n_rows)
    shots['row'], shots['col'] = zone_indices(shots['x'].values, shots['y'].values, n_cols, n_rows)

    move_counts = np.zeros((n_rows, n_cols))
    shot_counts = np.zeros((n_rows, n_cols))
    goal_counts = np.zeros((n_rows, n_cols))
    transition_counts = np.zeros((n_rows, n_cols, n_rows, n_cols))

    for (r, c), grp in moves.groupby(['row', 'col']):
        move_counts[r, c] = len(grp)
        dest_counts = grp.groupby(['row2', 'col2']).size()
        for (r2, c2), n in dest_counts.items():
            transition_counts[r, c, r2, c2] = n

    for (r, c), grp in shots.groupby(['row', 'col']):
        shot_counts[r, c] = len(grp)
        goal_counts[r, c] = grp['isGoal'].sum()

    total = move_counts + shot_counts
    move_prob = np.divide(move_counts, total, out=np.zeros_like(move_counts), where=total > 0)
    shoot_prob = np.divide(shot_counts, total, out=np.zeros_like(shot_counts), where=total > 0)
    goal_prob = np.divide(goal_counts, shot_counts, out=np.zeros_like(goal_counts), where=shot_counts > 0)
    transition_matrix = np.divide(
        transition_counts, move_counts[:, :, None, None],
        out=np.zeros_like(transition_counts),
        where=move_counts[:, :, None, None] > 0,
    )

    xt = np.zeros((n_rows, n_cols))
    for _ in range(n_iterations):
        move_component = np.einsum('ijkl,kl->ij', transition_matrix, xt)
        xt = shoot_prob * goal_prob + move_prob * move_component

    diagnostics = {
        'move_counts': move_counts, 'shot_counts': shot_counts,
        'goal_counts': goal_counts, 'move_prob': move_prob,
        'shoot_prob': shoot_prob, 'goal_prob': goal_prob,
    }
    return xt, diagnostics


def score_actions(moves: pd.DataFrame, xt_grid: np.ndarray,
                   n_rows: int = N_ROWS, n_cols: int = N_COLS) -> pd.DataFrame:
    moves = moves.copy()
    r1, c1 = zone_indices(moves['x'].values, moves['y'].values, n_cols, n_rows)
    r2, c2 = zone_indices(moves['endX'].values, moves['endY'].values, n_cols, n_rows)
    moves['xT'] = xt_grid[r2, c2] - xt_grid[r1, c1]
    return moves


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────

def save_grid(xt_grid: np.ndarray, n_moves: int, n_shots: int):
    np.savetxt(GRID_CSV_PATH, xt_grid, delimiter=',', fmt='%.6f')
    meta = {
        'fitted_at': datetime.now(timezone.utc).isoformat(),
        'n_rows': xt_grid.shape[0],
        'n_cols': xt_grid.shape[1],
        'n_iterations': N_ITERATIONS,
        'n_moves_fit_on': int(n_moves),
        'n_shots_fit_on': int(n_shots),
        'max_xt_value': float(xt_grid.max()),
    }
    META_JSON_PATH.write_text(json.dumps(meta, indent=2))


def load_grid() -> np.ndarray:
    return np.loadtxt(GRID_CSV_PATH, delimiter=',')


def write_xt_tables(scored_moves: pd.DataFrame, db_name: str = DB_PATH):
    """Full replace of xt_actions and xt — cheap to recompute from match_events."""
    conn = sqlite3.connect(db_name)
    try:
        actions_out = scored_moves[[
            'matchId', 'startDate', 'period', 'minute', 'second', 'teamId', 'playerId',
            'team', 'opponent', 'season', 'division', 'action_type', 'x', 'y', 'endX', 'endY', 'xT',
        ]]
        actions_out.to_sql('xt_actions', conn, if_exists='replace', index=False)

        grouped = scored_moves.groupby(['matchId', 'team', 'opponent', 'startDate', 'season', 'division'])['xT']
        agg = grouped.sum().reset_index()
        agg['gross_xT'] = grouped.apply(lambda s: s[s > 0].sum()).values
        agg.to_sql('xt', conn, if_exists='replace', index=False)

        print(f"✅ Wrote {len(actions_out)} rows to 'xt_actions', {len(agg)} rows to 'xt'")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def fit_and_write_xt(db_name: str = DB_PATH):
    print("📥 Loading match_events...")
    df = load_events(db_name)
    print(f"   {len(df):,} events loaded")

    print("⚙️  Building moves (passes + synthesized carries) and shots...")
    moves, shots = build_moves_and_shots(df)
    print(f"   {len(moves):,} moves ({(moves['action_type'] == 'pass').sum():,} passes, "
          f"{(moves['action_type'] == 'carry').sum():,} carries), {len(shots):,} shots")

    print(f"📐 Fitting {N_ROWS}x{N_COLS} xT grid ({N_ITERATIONS} iterations)...")
    xt_grid, _ = fit_xt_grid(moves, shots)
    save_grid(xt_grid, len(moves), len(shots))
    print(f"   Saved grid to {GRID_CSV_PATH.name} (max cell value {xt_grid.max():.4f})")

    print("🔢 Scoring actions...")
    scored_moves = score_actions(moves, xt_grid)

    print("💾 Writing xt_actions / xt tables...")
    write_xt_tables(scored_moves, db_name)

    return xt_grid, scored_moves


if __name__ == "__main__":
    fit_and_write_xt()
