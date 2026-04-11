"""
Premier League — weekly MAE evaluator.

Loads the full season once, then rolls forward week by week:
  - fits the model on all data known up to the prediction date
  - evaluates against actual scorelines in the following 7 days
  - tracks MAE on goals

Results are printed, written to weekly_accuracy_results.csv, and plotted.

Usage:
    python weekly_accuracy.py
"""
import os
import sys

NP_BAYES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NP_BAYES_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_utils import load_and_process_data
from src.model import build_and_sample_model
from manual_priors import MANUAL_ATT_PRIORS, MANUAL_DEF_PRIORS

# ── Config ───────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(NP_BAYES_DIR, '..', '..', '..', '..'))
DB_PATH   = os.path.join(REPO_ROOT, 'infra', 'data', 'db', 'fotmob.db')

LEAGUE = 'Premier_League'
SEASON = '2025-2026'

EVAL_START = '2025-09-01'   # first prediction date
EVAL_END   = '2026-04-01'   # last prediction date

# Small traces per week — speed vs accuracy trade-off
N_SAMPLES = 10_000
N_TUNE    = 500

# Full-season data load (no window — backtesting needs all history)
DECAY_RATE   = 0.0018
GOALS_WEIGHT = 0.25
XG_WEIGHT    = 0.50
PSXG_WEIGHT  = 0.15
EPV_WEIGHT   = 0.10
# ─────────────────────────────────────────────────────────────────────────────


def predict_week(df, prediction_date):
    """
    Fit model on all data before prediction_date.
    Evaluate on actual scorelines in [prediction_date, prediction_date + 7 days).
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

    # Consistent team mapping across the full season
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

    home_mu = np.mean(
        np.exp(base[rng, None] + att[rng][:, h_i] + defn[rng][:, a_i] + hadv[rng, None]),
        axis=0,
    )
    away_mu = np.mean(
        np.exp(base[rng, None] + att[rng][:, a_i] + defn[rng][:, h_i]),
        axis=0,
    )

    ah = test_df['home_goals'].values
    aa = test_df['away_goals'].values

    errors = np.abs(
        np.concatenate([home_mu, away_mu]) -
        np.concatenate([ah, aa])
    )

    return {
        'date':            prediction_date,
        'errors':          errors,          # raw per-prediction errors for pooled MAE
        'mae':             float(errors.mean()),
        'matches':         len(test_df),
        'home_actual':     float(np.mean(ah)),
        'away_actual':     float(np.mean(aa)),
        'total_actual':    float(np.mean(ah + aa)),
        'home_predicted':  float(np.mean(home_mu)),
        'away_predicted':  float(np.mean(away_mu)),
        'total_predicted': float(np.mean(home_mu + away_mu)),
    }


def main():
    print(f"Loading {LEAGUE} {SEASON} (full season)...")
    df, _, _ = load_and_process_data(
        db_path=DB_PATH,
        league=LEAGUE,
        season=SEASON,
        decay_rate=DECAY_RATE,
        goals_weight=GOALS_WEIGHT,
        xg_weight=XG_WEIGHT,
        psxg_weight=PSXG_WEIGHT,
    )

    dates   = pd.date_range(EVAL_START, EVAL_END, freq='W-TUE')
    records = []

    print(f"Running {len(dates)} weekly evaluations...")
    for d in dates:
        result = predict_week(df, d)
        if result:
            records.append(result)
            print(f"  {d.date()}  MAE={result['mae']:.3f}  n={result['matches']}")

    if not records:
        print("No results — check EVAL_START/EVAL_END against available data.")
        return

    results = pd.DataFrame(records)

    all_errors  = np.concatenate(results['errors'].values)
    pooled_mae  = float(all_errors.mean())
    total_preds = len(all_errors)

    print(f"\n=== SUMMARY ===")
    print(f"Weeks evaluated : {len(results)}")
    print(f"Predictions     : {total_preds}")
    print(f"MAE (pooled)    : {pooled_mae:.3f}")
    print(f"Home goals      : actual {results['home_actual'].mean():.2f}  "
          f"pred {results['home_predicted'].mean():.2f}")
    print(f"Away goals      : actual {results['away_actual'].mean():.2f}  "
          f"pred {results['away_predicted'].mean():.2f}")
    print(f"Total/game      : actual {results['total_actual'].mean():.2f}  "
          f"pred {results['total_predicted'].mean():.2f}")

    out = os.path.join(os.path.dirname(__file__), 'outputs', 'weekly_accuracy_results.csv')
    results.drop(columns=['errors']).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # mean_mae = results['mae'].mean()

    # ax1.plot(results['date'], results['mae'], 'bo-', linewidth=2)
    # ax1.axhline(mean_mae, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_mae:.3f}')
    # ax1.set_title('Weekly MAE — goals')
    # ax1.set_ylabel('MAE')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)

    # for col, lbl, sty in [
    #     ('home_actual',    'Home actual',     'bo-'),
    #     ('home_predicted', 'Home predicted',  'b--'),
    #     ('away_actual',    'Away actual',     'ro-'),
    #     ('away_predicted', 'Away predicted',  'r--'),
    #     ('total_actual',   'Total actual',    'ko-'),
    #     ('total_predicted','Total predicted', 'k--'),
    # ]:
    #     ax2.plot(results['date'], results[col], sty, label=lbl)

    # ax2.set_title('Goals per game: actual vs predicted')
    # ax2.set_ylabel('Goals per game')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
