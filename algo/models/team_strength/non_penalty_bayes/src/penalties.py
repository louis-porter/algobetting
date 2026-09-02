"""
penalties.py — penalty-goals addition for the non-penalty Bayesian goals models.

Two independent pieces:

- `get_penalty_baseline` — a per-league, recency-weighted penalty-AWARD rate (home/away),
  computed live from the DB each time it's called, rather than a frozen historical literal.
  Updates on its own as new matches get scraped in.

- `compute_penalty_multipliers` — a per-team multiplier on top of that baseline, from a
  fixed, historically-fitted relationship between a team's (non-penalty) goal difference
  relative to its league's current spread and its penalty rate. Good teams draw
  meaningfully more penalties than bad ones (quartile-by-quality teams roughly double their
  penalty rate from worst to best, consistently across three leagues and six seasons) — see
  analysis/penalties/penalty_prediction.ipynb and penalty_season_rate.ipynb for how
  QUALITY_SLOPE below was derived and why this only works as a *contemporaneous* adjustment
  (using a team's current-season strength estimate), not as a season-ahead forecast (team
  quality doesn't persist cleanly enough year to year for that — see the second notebook's
  in-season-updating section for why the naive forecasting version of this fails).

Both pieces are additive/multiplicative on top of BASELINE_HOME_PENS / BASELINE_AWAY_PENS in
each league's outputs.ipynb, in units of *expected goals from penalties* (i.e. already
multiplied by PENS_TO_GOALS) — matching how those constants were already used.
"""

import sqlite3

import numpy as np
import pandas as pd

# Historical penalty conversion rate (xG of a penalty kick) -- StatsBomb-style constant,
# already baked into BASELINE_HOME_PENS/AWAY_PENS wherever those are defined.
PENS_TO_GOALS = 0.78

# ~1-year half-life. Deliberately NOT the faster ~90-day decay used for team-level EMAs in
# the penalty analysis notebooks (analysis/penalties/) -- that's tuned for a reactive,
# per-team form signal. This is a *league-level* baseline meant to represent the
# competition's stable overall tendency (referee panel, competition style), so it should
# move slowly. Checked empirically: a 90-day half-life leaves an effective sample of only
# ~130-250 matches (mostly just the last couple of months) for this rate, which is far too
# reactive/noisy for a ~10-15% baseline rate. Matches each outputs.ipynb's own DECAY_RATE
# (0.0018) already used for the main goals model, for the same reason -- pass that constant
# in explicitly at the call site rather than relying on this default, so the two stay in sync
# if it's ever retuned.
DECAY_RATE = 0.0019

# Fitted once, offline: pens ~ Poisson(offset=log(games)), regressed on gd_z, where gd_z is
# a team-season's goal difference z-scored *within that season's own league* (so promoted/
# relegated teams are comparable). Pooled across Premier League / Championship / Superligaen
# team-seasons, 2020-21 through 2025-26 (n=326). Coefficient 0.228, z=8.6, p<0.001 -- see
# penalty_season_rate.ipynb. This is a structural relationship (how much quality translates
# into extra penalties) rather than a specific team's quality level, so unlike that notebook's
# season-ahead forecasts, it's expected to generalize across seasons.
QUALITY_SLOPE = 0.228


def get_penalty_baseline(db_path, league, as_of=None, decay_rate=DECAY_RATE):
    """Recency-weighted penalty-award rate (fraction of matches with >=1 penalty for that
    side), split home/away, for `league`, using every played match in the DB up to `as_of`
    (defaults to now). Returns raw award-rate, NOT yet converted to expected goals --
    multiply by PENS_TO_GOALS at the call site, matching how the old literal constants
    (e.g. `0.157 * 0.78`) were expressed.

    Returns (0.0, 0.0) if there's no history yet for this league (shouldn't happen in
    practice, but avoids a division-by-zero on an empty/new league).
    """
    conn = sqlite3.connect(db_path)
    try:
        m = pd.read_sql_query(
            """
            SELECT m.match_id, m.match_date, p.home_pens, p.away_pens
            FROM matches m
            LEFT JOIN penalties p ON p.match_id = m.match_id
            WHERE m.league_id = ? AND m.home_goals IS NOT NULL
            """,
            conn, params=[league],
        )
    finally:
        conn.close()

    if m.empty:
        return 0.0, 0.0

    m[['home_pens', 'away_pens']] = m[['home_pens', 'away_pens']].fillna(0)
    m['match_date'] = pd.to_datetime(m['match_date'])

    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    m = m[m['match_date'] <= as_of_ts]
    if m.empty:
        return 0.0, 0.0

    days = (as_of_ts - m['match_date']).dt.days.clip(lower=0).to_numpy(dtype=float)
    weights = np.exp(-decay_rate * days)

    home_rate = float(np.average((m['home_pens'] > 0).astype(float), weights=weights))
    away_rate = float(np.average((m['away_pens'] > 0).astype(float), weights=weights))
    return home_rate, away_rate


def compute_penalty_multipliers(team_mapping, att_samples, def_samples, base_samples,
                                 hadv_samples, slope=QUALITY_SLOPE):
    """Per-team multiplier on the league's flat penalty baseline, from each team's own
    (non-penalty) round-robin goal difference against every other team currently in the
    league -- the same quantity already used elsewhere (e.g. `ratings_df`/`avg_table`) to
    characterize team quality, computed here *before* any penalty addition to avoid
    circularity (using a number that already includes a flat penalty guess to decide how
    to correct the flat guess).

    Renormalized so the multipliers average exactly 1 across the league's current teams --
    this only redistributes the already-calibrated league-wide baseline by relative
    quality, it doesn't change the league's overall average penalty rate.

    Parameters mirror the posterior arrays already extracted in each outputs.ipynb
    (e.g. `att = posterior['att_str'].values.reshape(-1, n_teams)`): each of
    `att_samples`/`def_samples` is (n_draws, n_teams), `base_samples`/`hadv_samples` is
    (n_draws,).

    Returns {team_name: multiplier}.
    """
    teams = list(team_mapping.keys())
    goals_for = {t: 0.0 for t in teams}
    goals_against = {t: 0.0 for t in teams}
    matches = {t: 0 for t in teams}

    for home in teams:
        hi = team_mapping[home]
        for away in teams:
            if home == away:
                continue
            ai = team_mapping[away]
            home_lam = np.exp(base_samples + hadv_samples + att_samples[:, hi] + def_samples[:, ai]).mean()
            away_lam = np.exp(base_samples + att_samples[:, ai] + def_samples[:, hi]).mean()

            goals_for[home] += home_lam
            goals_against[home] += away_lam
            matches[home] += 1
            goals_for[away] += away_lam
            goals_against[away] += home_lam
            matches[away] += 1

    goal_diff = pd.Series({
        t: (goals_for[t] - goals_against[t]) / matches[t] if matches[t] else 0.0
        for t in teams
    })

    std = goal_diff.std(ddof=0)
    z = (goal_diff - goal_diff.mean()) / std if std > 0 else goal_diff * 0.0

    raw_multiplier = np.exp(slope * z)
    multiplier = raw_multiplier / raw_multiplier.mean()
    return multiplier.to_dict()
