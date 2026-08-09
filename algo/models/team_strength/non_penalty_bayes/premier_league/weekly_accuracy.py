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

    # Look up full actual scores (incl. penalties) and xG from matches table
    test_df = test_df.merge(
        actual_df[['match_id', 'home_goals', 'away_goals', 'home_xg', 'away_xg']],
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

    # Full actual scores including penalties, and xG
    ah   = test_df['home_goals'].values
    aa   = test_df['away_goals'].values
    hxg  = test_df['home_xg'].values
    axg  = test_df['away_xg'].values

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
            'home_xg':        float(hxg[i]) if not np.isnan(hxg[i]) else None,
            'away_xg':        float(axg[i]) if not np.isnan(axg[i]) else None,
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
    """Load full scorelines (incl. penalties) and xG from the matches table."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            m.match_id,
            m.home_goals,
            m.away_goals,
            CAST(ms.home_expected_goals AS REAL) AS home_xg,
            CAST(ms.away_expected_goals AS REAL) AS away_xg
        FROM matches m
        LEFT JOIN match_stats ms ON ms.match_id = m.match_id
        WHERE m.league_id = ? AND m.season = ?
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


def save_to_db(db_path, league, season, records, team_stats):
    """
    Write prediction results to fotmob.db.

    Tables created (if missing):
      pl_match_predictions  — one row per match
      pl_gw_accuracy        — one row per gameweek summary

    Existing rows for this league+season are deleted before inserting so
    re-running the script always gives a clean, up-to-date snapshot.
    """
    from datetime import datetime
    run_ts = datetime.utcnow().isoformat(timespec='seconds')

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # ── match-level predictions ───────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pl_match_predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp   TEXT,
            league          TEXT,
            season          TEXT,
            gw              INTEGER,
            gw_start        TEXT,
            gw_end          TEXT,
            home_team       TEXT,
            away_team       TEXT,
            home_predicted  REAL,
            away_predicted  REAL,
            home_actual     REAL,
            away_actual     REAL,
            home_error      REAL,
            away_error      REAL,
            home_xg         REAL,
            away_xg         REAL,
            home_xg_error   REAL,
            away_xg_error   REAL
        )
    """)
    cur.execute(
        "DELETE FROM pl_match_predictions WHERE league = ? AND season = ?",
        (league, season),
    )
    match_rows_db = []
    for r in records:
        for m in r['match_rows']:
            hxg = m.get('home_xg')
            axg = m.get('away_xg')
            match_rows_db.append((
                run_ts, league, season,
                r['gw'], str(r['gw_start']), str(r['gw_end']),
                m['home_team'],      m['away_team'],
                m['home_predicted'], m['away_predicted'],
                m['home_actual'],    m['away_actual'],
                abs(m['home_predicted'] - m['home_actual']),
                abs(m['away_predicted'] - m['away_actual']),
                hxg, axg,
                abs(m['home_predicted'] - hxg) if hxg is not None else None,
                abs(m['away_predicted'] - axg) if axg is not None else None,
            ))
    cur.executemany("""
        INSERT INTO pl_match_predictions
            (run_timestamp, league, season, gw, gw_start, gw_end,
             home_team, away_team,
             home_predicted, away_predicted,
             home_actual, away_actual,
             home_error, away_error,
             home_xg, away_xg,
             home_xg_error, away_xg_error)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, match_rows_db)

    # ── gameweek summary ──────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pl_gw_accuracy (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp    TEXT,
            league           TEXT,
            season           TEXT,
            gw               INTEGER,
            gw_start         TEXT,
            gw_end           TEXT,
            matches          INTEGER,
            mae              REAL,
            home_actual      REAL,
            away_actual      REAL,
            total_actual     REAL,
            home_predicted   REAL,
            away_predicted   REAL,
            total_predicted  REAL
        )
    """)
    cur.execute(
        "DELETE FROM pl_gw_accuracy WHERE league = ? AND season = ?",
        (league, season),
    )
    gw_rows_db = [
        (
            run_ts, league, season,
            r['gw'], str(r['gw_start']), str(r['gw_end']),
            r['matches'], r['mae'],
            r['home_actual'], r['away_actual'], r['total_actual'],
            r['home_predicted'], r['away_predicted'], r['total_predicted'],
        )
        for r in records
    ]
    cur.executemany("""
        INSERT INTO pl_gw_accuracy
            (run_timestamp, league, season, gw, gw_start, gw_end,
             matches, mae,
             home_actual, away_actual, total_actual,
             home_predicted, away_predicted, total_predicted)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, gw_rows_db)

    conn.commit()
    conn.close()

    n_matches = len(match_rows_db)
    n_gws     = len(gw_rows_db)
    print(f"\nSaved to DB ({db_path}):")
    print(f"  pl_match_predictions  → {n_matches} rows")
    print(f"  pl_gw_accuracy        → {n_gws} rows")


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

    # ── Write to DB ───────────────────────────────────────────────────────────
    save_to_db(DB_PATH, LEAGUE, SEASON, records, team_stats)

    # ── Save CSVs (kept for backwards compat) ─────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    weekly_out = os.path.join(out_dir, 'weekly_accuracy_results.csv')
    results.drop(columns=['errors', 'match_rows']).to_csv(weekly_out, index=False)
    print(f"Saved: {weekly_out}")

    team_out = os.path.join(out_dir, 'team_accuracy_results.csv')
    team_stats.to_csv(team_out, index=False)
    print(f"Saved: {team_out}")


if __name__ == '__main__':
    main()
