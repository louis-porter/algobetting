"""
Premier League — per-gameweek MAE evaluator.

Uses custom_gw from the DB to define evaluation windows, so each window
maps exactly to one gameweek (Fri-Mon or Tue-Thu) with no team appearing
twice in the same window.

Results are printed and written to weekly_accuracy_results.csv.

Usage:
    python weekly_accuracy.py
"""
import os
import sys

NP_BAYES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NP_BAYES_DIR)

import sqlite3
import numpy as np
import pandas as pd

from src.data_utils import load_and_process_data
from src.model import build_and_sample_model
from manual_priors import MANUAL_ATT_PRIORS, MANUAL_DEF_PRIORS

# ── Config ───────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(NP_BAYES_DIR, '..', '..', '..', '..'))
DB_PATH   = os.path.join(REPO_ROOT, 'infra', 'data', 'db', 'fotmob.db')

LEAGUE = 'Premier_League'
SEASON = '2025-2026'

EVAL_START = '2025-09-22'
EVAL_END   = '2026-05-30'

N_SAMPLES = 20_000
N_TUNE    = 10_000

DECAY_RATE   = 0.0018
GOALS_WEIGHT = 0.25
XG_WEIGHT    = 0.50
PSXG_WEIGHT  = 0.15
EPV_WEIGHT   = 0.10

# Penalty baseline — matches outputs.ipynb
BASELINE_HOME_PENS = 0.157 * 0.78
BASELINE_AWAY_PENS = 0.101 * 0.78
# ─────────────────────────────────────────────────────────────────────────────


def predict_gw(df, actual_df, gw, gw_start, gw_end):
    """
    Fit model on all data strictly before gw_start.
    Evaluate on actual scorelines (incl. penalties) within [gw_start, gw_end].
    Returns dict of metrics, or None if no test matches in the window.
    """
    df = df.copy()
    df['match_date'] = pd.to_datetime(df['match_date'])
    gw_start = pd.to_datetime(gw_start)
    gw_end   = pd.to_datetime(gw_end)

    train_df = df[df['match_date'] < gw_start].copy()
    test_df  = df[
        (df['match_date'] >= gw_start) &
        (df['match_date'] <= gw_end) &
        (df['is_actual']  == True)
    ].copy()

    if len(test_df) == 0 or len(train_df) == 0:
        return None

    # Look up full actual scores (incl. penalties) from matches table
    test_df = test_df.merge(
        actual_df[['match_id', 'home_goals', 'away_goals']],
        on='match_id', suffixes=('_np', ''),
    )

    all_teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
    team_map  = {t: i for i, t in enumerate(all_teams)}
    n_teams   = len(all_teams)

    train_df['home_idx'] = train_df['home_team'].map(team_map)
    train_df['away_idx'] = train_df['away_team'].map(team_map)
    test_df['home_idx']  = test_df['home_team'].map(team_map)
    test_df['away_idx']  = test_df['away_team'].map(team_map)

    _, trace = build_and_sample_model(
        train_df, n_teams,
        trace=N_SAMPLES, tune=N_TUNE,
        manual_att_priors=MANUAL_ATT_PRIORS,
        manual_def_priors=MANUAL_DEF_PRIORS,
        team_mapping=team_map,
    )

    posterior = trace.posterior
    att  = posterior['att_str'].values.reshape(-1, n_teams)
    defn = posterior['def_str'].values.reshape(-1, n_teams)
    hadv = posterior['home_adv'].values.reshape(-1)
    base = posterior['baseline'].values.reshape(-1)

    rng = np.random.choice(len(base), size=500, replace=True)
    h_i = test_df['home_idx'].values
    a_i = test_df['away_idx'].values

    # Non-penalty predictions + penalty baseline
    home_mu = np.mean(
        np.exp(base[rng, None] + att[rng][:, h_i] + defn[rng][:, a_i] + hadv[rng, None]),
        axis=0,
    ) + BASELINE_HOME_PENS
    away_mu = np.mean(
        np.exp(base[rng, None] + att[rng][:, a_i] + defn[rng][:, h_i]),
        axis=0,
    ) + BASELINE_AWAY_PENS

    # Full actual scores including penalties
    ah = test_df['home_goals'].values
    aa = test_df['away_goals'].values

    errors = np.abs(
        np.concatenate([home_mu, away_mu]) -
        np.concatenate([ah, aa])
    )

    match_rows = [
        {
            'gw':             gw,
            'home_team':      test_df['home_team'].iloc[i],
            'away_team':      test_df['away_team'].iloc[i],
            'home_predicted': float(home_mu[i]),
            'away_predicted': float(away_mu[i]),
            'home_actual':    float(ah[i]),
            'away_actual':    float(aa[i]),
        }
        for i in range(len(test_df))
    ]

    return {
        'gw':              gw,
        'gw_start':        gw_start.date(),
        'gw_end':          gw_end.date(),
        'errors':          errors,
        'mae':             float(errors.mean()),
        'matches':         len(test_df),
        'home_actual':     float(np.mean(ah)),
        'away_actual':     float(np.mean(aa)),
        'total_actual':    float(np.mean(ah + aa)),
        'home_predicted':  float(np.mean(home_mu)),
        'away_predicted':  float(np.mean(away_mu)),
        'total_predicted': float(np.mean(home_mu + away_mu)),
        'match_rows':      match_rows,
    }


def build_team_breakdown(records):
    """Per-team scored and conceded accuracy."""
    rows = [m for r in records for m in r['match_rows']]
    df   = pd.DataFrame(rows)

    home = pd.DataFrame({
        'team':            df['home_team'].values,
        'sc_actual':       df['home_actual'].values,
        'sc_pred':         df['home_predicted'].values,
        'co_actual':       df['away_actual'].values,
        'co_pred':         df['away_predicted'].values,
    })
    away = pd.DataFrame({
        'team':            df['away_team'].values,
        'sc_actual':       df['away_actual'].values,
        'sc_pred':         df['away_predicted'].values,
        'co_actual':       df['home_actual'].values,
        'co_pred':         df['home_predicted'].values,
    })

    combined = pd.concat([home, away], ignore_index=True)
    combined['err_sc'] = (combined['sc_pred'] - combined['sc_actual']).abs()
    combined['err_co'] = (combined['co_pred'] - combined['co_actual']).abs()

    team_stats = (
        combined.groupby('team')
        .agg(
            matches    = ('err_sc', 'count'),
            sc_actual  = ('sc_actual', 'mean'),
            sc_pred    = ('sc_pred',   'mean'),
            mae_sc     = ('err_sc',    'mean'),
            co_actual  = ('co_actual', 'mean'),
            co_pred    = ('co_pred',   'mean'),
            mae_co     = ('err_co',    'mean'),
        )
        .reset_index()
    )
    team_stats['mae_overall'] = (team_stats['mae_sc'] + team_stats['mae_co']) / 2
    return team_stats.sort_values('mae_overall').reset_index(drop=True)


def print_team_table(team_stats):
    w = 62
    print(f"\n{'─'*w}")
    print("  PER TEAM  (sorted by overall MAE, best → worst)")
    print(f"{'─'*w}")
    hdr = (f"  {'Team':<26} {'M':>3}  "
           f"{'Sc-Act':>6}  {'Sc-Pred':>7}  {'MAE-Sc':>6}  "
           f"{'Co-Act':>6}  {'Co-Pred':>7}  {'MAE-Co':>6}  "
           f"{'MAE-Ov':>6}")
    print(hdr)
    print(f"  {'─'*len(hdr.strip())}")
    for _, row in team_stats.iterrows():
        print(
            f"  {row['team']:<26} {int(row['matches']):>3}  "
            f"{row['sc_actual']:>6.2f}  {row['sc_pred']:>7.2f}  {row['mae_sc']:>6.3f}  "
            f"{row['co_actual']:>6.2f}  {row['co_pred']:>7.2f}  {row['mae_co']:>6.3f}  "
            f"{row['mae_overall']:>6.3f}"
        )
    print(f"{'─'*w}")


def print_summary(records):
    """Overall accuracy from match-level data (not average of GW averages)."""
    rows = [m for r in records for m in r['match_rows']]
    df   = pd.DataFrame(rows)

    home_err = (df['home_predicted'] - df['home_actual']).abs()
    away_err = (df['away_predicted'] - df['away_actual']).abs()
    all_err  = pd.concat([home_err, away_err])

    w = 62
    print(f"\n{'─'*w}")
    print(f"  OVERALL  —  {len(records)} gameweeks  ·  {len(df)} matches  ·  {len(all_err)} predictions")
    print(f"{'─'*w}")
    tot_err = ((df['home_predicted'] + df['away_predicted']) - (df['home_actual'] + df['away_actual'])).abs()

    print(f"  {'MAE (all predictions)':<26} {all_err.mean():.4f}")
    print(f"  {'MAE home':<26} {home_err.mean():.4f}")
    print(f"  {'MAE away':<26} {away_err.mean():.4f}")
    print(f"  {'MAE total goals (per game)':<26} {tot_err.mean():.4f}")
    print(f"  {'Home goals':<22} actual {df['home_actual'].mean():.2f}   pred {df['home_predicted'].mean():.2f}")
    print(f"  {'Away goals':<22} actual {df['away_actual'].mean():.2f}   pred {df['away_predicted'].mean():.2f}")
    tot_act  = (df['home_actual']    + df['away_actual']).mean()
    tot_pred = (df['home_predicted'] + df['away_predicted']).mean()
    print(f"  {'Total / game':<22} actual {tot_act:.2f}   pred {tot_pred:.2f}")
    print(f"{'─'*w}")


def print_weekly_table(records):
    """One row per gameweek."""
    hdr = (f"  {'GW':>3}  {'Start':>10}  {'End':>10}  "
           f"{'n':>3}  {'MAE':>6}  "
           f"{'H-act':>5}  {'H-pred':>6}  "
           f"{'A-act':>5}  {'A-pred':>6}  "
           f"{'Tot-act':>7}  {'Tot-pred':>8}")
    w = 62
    print(f"\n{'─'*w}")
    print("  PER GAMEWEEK")
    print(f"{'─'*w}")
    print(hdr)
    print(f"  {'─'*len(hdr.strip())}")
    for r in records:
        mdf = pd.DataFrame(r['match_rows'])
        print(
            f"  {r['gw']:>3}  {str(r['gw_start']):>10}  {str(r['gw_end']):>10}  "
            f"{r['matches']:>3}  {r['mae']:>6.3f}  "
            f"{mdf['home_actual'].mean():>5.2f}  {mdf['home_predicted'].mean():>6.2f}  "
            f"{mdf['away_actual'].mean():>5.2f}  {mdf['away_predicted'].mean():>6.2f}  "
            f"{(mdf['home_actual']+mdf['away_actual']).mean():>7.2f}  "
            f"{(mdf['home_predicted']+mdf['away_predicted']).mean():>8.2f}"
        )
    print(f"{'─'*w}")


def load_actual_full_scores(db_path, league, season):
    """Load full scorelines (incl. penalties) from the matches table."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            match_id,
            home_goals,
            away_goals
        FROM matches
        WHERE league_id = ? AND season = ?
    """, conn, params=[league, season])
    conn.close()
    return df


def load_gw_windows(db_path, league, season, eval_start, eval_end):
    """Return a DataFrame of custom GW windows within the evaluation range."""
    conn = sqlite3.connect(db_path)
    gw_df = pd.read_sql("""
        SELECT custom_gw, MIN(match_date) AS gw_start, MAX(match_date) AS gw_end
        FROM matches
        WHERE league_id = ? AND season = ? AND home_goals IS NOT NULL
          AND custom_gw IS NOT NULL
        GROUP BY custom_gw
        ORDER BY custom_gw
    """, conn, params=[league, season])
    conn.close()

    gw_df = gw_df[
        (gw_df['gw_start'] >= eval_start) &
        (gw_df['gw_end']   <= eval_end)
    ].reset_index(drop=True)
    return gw_df


def main():
    actual_df = load_actual_full_scores(DB_PATH, LEAGUE, SEASON)
    print(f"Loaded {len(actual_df)} full-score matches from matches table.")

    print("Loading dataset...")
    df, _, _ = load_and_process_data(
        db_path=DB_PATH, league=LEAGUE, season=SEASON,
        decay_rate=DECAY_RATE,
        goals_weight=GOALS_WEIGHT, xg_weight=XG_WEIGHT,
        psxg_weight=PSXG_WEIGHT, epv_weight=EPV_WEIGHT,
    )

    gw_windows = load_gw_windows(DB_PATH, LEAGUE, SEASON, EVAL_START, EVAL_END)
    records = []

    print(f"\nRunning {len(gw_windows)} gameweek evaluations (GW windows from DB)...")
    for _, row in gw_windows.iterrows():
        r = predict_gw(df, actual_df, row['custom_gw'], row['gw_start'], row['gw_end'])
        if r:
            records.append(r)
            print(f"  GW{r['gw']:2d}  {r['gw_start']} → {r['gw_end']}  MAE={r['mae']:.3f}  n={r['matches']}")

    if not records:
        print("No results — check EVAL_START/EVAL_END against available data.")
        return

    results    = pd.DataFrame(records)
    team_stats = build_team_breakdown(records)

    print_summary(records)
    print_weekly_table(records)
    print_team_table(team_stats)

    # Save CSVs
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    weekly_out = os.path.join(out_dir, 'weekly_accuracy_results.csv')
    results.drop(columns=['errors', 'match_rows']).to_csv(weekly_out, index=False)
    print(f"\nSaved: {weekly_out}")

    team_out = os.path.join(out_dir, 'team_accuracy_results.csv')
    team_stats.to_csv(team_out, index=False)
    print(f"Saved: {team_out}")


if __name__ == '__main__':
    main()
