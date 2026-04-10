"""
Superligaen — non-penalty Bayesian team strength model.
Loads full-season data and fits the model. Run this to get a trace object,
or just open outputs.ipynb which runs the model inline.

Usage:
    python run_model.py
"""
import os
import sys

NP_BAYES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NP_BAYES_DIR)

from src.data_utils import load_and_process_data
from src.superliga_model import build_and_sample_model

# ── Config ───────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(NP_BAYES_DIR, '..', '..', '..'))
DB_PATH   = os.path.join(REPO_ROOT, 'infra', 'data', 'db', 'fotmob.db')

LEAGUE = 'Superligaen'
SEASON = '2025-2026'

DECAY_RATE   = 0.0018
GOALS_WEIGHT = 0.27
XG_WEIGHT    = 0.55
PSXG_WEIGHT  = 0.17
EPV_WEIGHT   = 0.0

N_SAMPLES = 2_000
N_TUNE    = 1_000
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"Loading {LEAGUE} {SEASON}...")
    df, team_mapping, n_teams = load_and_process_data(
        db_path=DB_PATH,
        league=LEAGUE,
        season=SEASON,
        decay_rate=DECAY_RATE,
        goals_weight=GOALS_WEIGHT,
        xg_weight=XG_WEIGHT,
        psxg_weight=PSXG_WEIGHT,
        epv_weight=EPV_WEIGHT,
    )
    print(f"  {n_teams} teams, {df['match_id'].nunique()} matches")

    print("Sampling model...")
    _, trace = build_and_sample_model(
        df, n_teams,
        trace=N_SAMPLES,
        tune=N_TUNE,
        team_mapping=team_mapping,
    )
    print("Done.")
    return trace, team_mapping, n_teams


if __name__ == '__main__':
    main()
