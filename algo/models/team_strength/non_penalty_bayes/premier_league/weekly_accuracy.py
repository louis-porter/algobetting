"""
Premier League — weekly MAE evaluator.

Runs two variants side-by-side every week:
  BASE  — normal weights, no garbage-time adjustment
  GC    — same weights applied to competitive-minutes-only signals

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

EVAL_START = '2025-09-01'
EVAL_END   = '2026-04-01'

N_SAMPLES = 10_000
N_TUNE    = 5000

DECAY_RATE   = 0.0018
GOALS_WEIGHT = 0.25
XG_WEIGHT    = 0.50
PSXG_WEIGHT  = 0.15
EPV_WEIGHT   = 0.10

# Penalty baseline — matches outputs.ipynb
BASELINE_HOME_PENS = 0.157 * 0.78
BASELINE_AWAY_PENS = 0.101 * 0.78
# ─────────────────────────────────────────────────────────────────────────────


def predict_week(df, actual_df, prediction_date):
    """
    Fit model on all data before prediction_date.
    Evaluate on actual scorelines (incl. penalties) in [prediction_date, prediction_date + 7 days).
    Returns dict of metrics, or None if no test matches that week.
    """
    df = df.copy()
    df['match_date'] = pd.to_datetime(df['match_date'])
    prediction_date  = pd.to_datetime(prediction_date)

    train_df = df[df['match_date'] < prediction_date].copy()
    test_df  = df[
        (df['match_date'] >= prediction_date) &
        (df['match_date'] <  prediction_date + pd.Timedelta(days=7)) &
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

    return {
        'date':            prediction_date,
        'errors':          errors,
        'mae':             float(errors.mean()),
        'matches':         len(test_df),
        'home_actual':     float(np.mean(ah)),
        'away_actual':     float(np.mean(aa)),
        'total_actual':    float(np.mean(ah + aa)),
        'home_predicted':  float(np.mean(home_mu)),
        'away_predicted':  float(np.mean(away_mu)),
        'total_predicted': float(np.mean(home_mu + away_mu)),
    }


def print_summary(label, results):
    all_errors = np.concatenate(results['errors'].values)
    pooled_mae = float(all_errors.mean())
    print(f"\n=== {label} ===")
    print(f"Weeks evaluated : {len(results)}")
    print(f"Predictions     : {len(all_errors)}")
    print(f"MAE (pooled)    : {pooled_mae:.4f}")
    print(f"Home goals      : actual {results['home_actual'].mean():.2f}  "
          f"pred {results['home_predicted'].mean():.2f}")
    print(f"Away goals      : actual {results['away_actual'].mean():.2f}  "
          f"pred {results['away_predicted'].mean():.2f}")
    print(f"Total/game      : actual {results['total_actual'].mean():.2f}  "
          f"pred {results['total_predicted'].mean():.2f}")
    return pooled_mae


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


def main():
    actual_df = load_actual_full_scores(DB_PATH, LEAGUE, SEASON)
    print(f"Loaded {len(actual_df)} full-score matches from matches table.")

    shared_kwargs = dict(
        db_path=DB_PATH,
        league=LEAGUE,
        season=SEASON,
        decay_rate=DECAY_RATE,
    )

    print("Loading dataset...")
    df, _, _ = load_and_process_data(
        **shared_kwargs,
        goals_weight=GOALS_WEIGHT,
        xg_weight=XG_WEIGHT,
        psxg_weight=PSXG_WEIGHT,
        epv_weight=EPV_WEIGHT,
    )

    dates   = pd.date_range(EVAL_START, EVAL_END, freq='W-TUE')
    records = []

    print(f"\nRunning {len(dates)} weekly evaluations...")
    for d in dates:
        r = predict_week(df, actual_df, d)
        if r:
            records.append(r)
            print(f"  {d.date()}  MAE={r['mae']:.3f}  n={r['matches']}")

    if not records:
        print("No results — check EVAL_START/EVAL_END against available data.")
        return

    results = pd.DataFrame(records)
    print_summary("Model", results)

    # Save CSV
    out = os.path.join(os.path.dirname(__file__), 'outputs', 'weekly_accuracy_results.csv')
    results.drop(columns=['errors']).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
