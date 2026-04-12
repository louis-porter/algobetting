"""
simulation.py — season simulation and helper utilities for the non-penalty Bayesian model.

Key functions
─────────────
run_multiple_seasons     Full Monte-Carlo season simulation
precompute_expected_goals Pre-compute posterior-mean xG for every fixture
predict_match            Draw match probabilities / xG from the posterior
simulate_full_season_fast Single fast simulation pass
load_actual_results      Load played match results from the DB
get_actual_standings     Current league table from actual_results dict
form_net_rating          Weighted form rating from scoreline data (no model needed)
get_last_result          Last result string for a team
get_form_string          W/D/L form string
rank_arrow               ▲ / ▼ / ▬ movement indicator
"""

import sqlite3
import numpy as np
import pandas as pd

BASELINE_HOME_PENS = 0.157 * 0.78
BASELINE_AWAY_PENS = 0.101 * 0.78


# ── Match prediction ──────────────────────────────────────────────────────────

def predict_match(home_team, away_team, trace, team_mapping,
                  home_pen_rate=BASELINE_HOME_PENS,
                  away_pen_rate=BASELINE_AWAY_PENS):
    hi = team_mapping[home_team]
    ai = team_mapping[away_team]
    n  = len(team_mapping)

    att  = trace.posterior['att_str'].values.reshape(-1, n)
    defn = trace.posterior['def_str'].values.reshape(-1, n)
    base = trace.posterior['baseline'].values.flatten()
    hadv = trace.posterior['home_adv'].values.flatten()

    h_lam = np.exp(base + hadv + att[:, hi] + defn[:, ai]) + home_pen_rate
    a_lam = np.exp(base         + att[:, ai] + defn[:, hi]) + away_pen_rate

    hg = np.random.poisson(h_lam)
    ag = np.random.poisson(a_lam)

    return {
        'home_goals_expected': float(np.mean(h_lam)),
        'away_goals_expected': float(np.mean(a_lam)),
        'home_win_prob':       float(np.mean(hg > ag)),
        'draw_prob':           float(np.mean(hg == ag)),
        'away_win_prob':       float(np.mean(hg < ag)),
    }


# ── Expected-goals pre-computation ───────────────────────────────────────────

def precompute_expected_goals(trace, team_mapping, df_actual,
                              home_pen_rate=BASELINE_HOME_PENS,
                              away_pen_rate=BASELINE_AWAY_PENS):
    """
    Returns
    -------
    actual_results : dict  (home, away) → (hg, ag, h_xg, a_xg)
    expected_goals : dict  (home, away) → (h_xg, a_xg)   for unplayed fixtures
    """
    teams = list(team_mapping.keys())
    actual_results = {}
    expected_goals = {}

    played = {}
    if df_actual is not None and 'is_actual' in df_actual.columns:
        for _, row in df_actual[df_actual['is_actual']].iterrows():
            played[(row['home_team'], row['away_team'])] = row

    print(f"Pre-computing xG for {len(teams) * (len(teams)-1)} fixtures...")
    for home in teams:
        for away in teams:
            if home == away:
                continue
            key  = (home, away)
            pred = predict_match(home, away, trace, team_mapping,
                                 home_pen_rate, away_pen_rate)
            hxg, axg = pred['home_goals_expected'], pred['away_goals_expected']

            if key in played:
                row = played[key]
                hg  = int(row['home_goals']) if pd.notna(row['home_goals']) else None
                ag  = int(row['away_goals']) if pd.notna(row['away_goals']) else None
                if hg is not None and ag is not None:
                    actual_results[key] = (hg, ag, hxg, axg)
                else:
                    expected_goals[key] = (hxg, axg)
            else:
                expected_goals[key] = (hxg, axg)

    print(f"  Played: {len(actual_results)}   To simulate: {len(expected_goals)}")
    return actual_results, expected_goals


# ── Single simulation pass ────────────────────────────────────────────────────

def _update_table(t, ht, at, hg, ag, hxg, axg):
    t[ht]['played'] += 1;  t[at]['played'] += 1
    t[ht]['gf'] += hg;     t[ht]['ga'] += ag
    t[at]['gf'] += ag;     t[at]['ga'] += hg
    t[ht]['xgf'] += hxg;   t[ht]['xga'] += axg
    t[at]['xgf'] += axg;   t[at]['xga'] += hxg
    if hg > ag:
        t[ht]['pts'] += 3; t[ht]['w'] += 1; t[at]['l'] += 1
    elif hg == ag:
        t[ht]['pts'] += 1; t[ht]['d'] += 1
        t[at]['pts'] += 1; t[at]['d'] += 1
    else:
        t[at]['pts'] += 3; t[at]['w'] += 1; t[ht]['l'] += 1


def simulate_full_season_fast(actual_results, expected_goals, teams):
    tbl = {t: {'played':0,'w':0,'d':0,'l':0,'gf':0,'ga':0,'xgf':0.0,'xga':0.0,'pts':0}
           for t in teams}

    unplayed = list(expected_goals.keys())
    if unplayed:
        hxg_arr = np.array([expected_goals[k][0] for k in unplayed])
        axg_arr = np.array([expected_goals[k][1] for k in unplayed])
        hg_sim  = np.random.poisson(hxg_arr)
        ag_sim  = np.random.poisson(axg_arr)

    for (ht, at), (hg, ag, hxg, axg) in actual_results.items():
        _update_table(tbl, ht, at, hg, ag, hxg, axg)

    for i, (ht, at) in enumerate(unplayed):
        hxg, axg = expected_goals[(ht, at)]
        _update_table(tbl, ht, at, hg_sim[i], ag_sim[i], hxg, axg)

    for t in teams:
        tbl[t]['gd']  = tbl[t]['gf']  - tbl[t]['ga']
        tbl[t]['xgd'] = tbl[t]['xgf'] - tbl[t]['xga']
    return tbl


# ── Full Monte-Carlo simulation ───────────────────────────────────────────────

def run_multiple_seasons(n_simulations, trace, team_mapping, df_actual,
                         home_pen_rate=BASELINE_HOME_PENS,
                         away_pen_rate=BASELINE_AWAY_PENS):
    teams   = list(team_mapping.keys())
    n       = len(teams)

    actual_results, expected_goals = precompute_expected_goals(
        trace, team_mapping, df_actual, home_pen_rate, away_pen_rate)

    acc = {k: np.zeros(n, dtype=np.float64) for k in
           ['pts', 'pts_sq', 'w', 'd', 'l', 'gf', 'ga', 'xgf', 'xga', 'pos']}
    cnt = {k: np.zeros(n, dtype=np.int32) for k in
           ['title', 'top5', 'top8', 'rel']}
    pos_freq = np.zeros((n, n), dtype=np.int32)

    print(f"Running {n_simulations:,} simulations...")
    for sim in range(n_simulations):
        if sim % 2000 == 0 and sim > 0:
            print(f"  {sim:,} / {n_simulations:,}")
        tbl   = simulate_full_season_fast(actual_results, expected_goals, teams)
        pts   = np.array([tbl[t]['pts'] for t in teams])
        gd    = np.array([tbl[t]['gd']  for t in teams])
        gf    = np.array([tbl[t]['gf']  for t in teams])
        order = np.lexsort((gf, gd, pts))[::-1]
        positions = np.empty(n, dtype=np.int32)
        positions[order] = np.arange(1, n + 1)

        acc['pts']    += pts
        acc['pts_sq'] += pts ** 2
        acc['w']      += [tbl[t]['w']   for t in teams]
        acc['d']      += [tbl[t]['d']   for t in teams]
        acc['l']      += [tbl[t]['l']   for t in teams]
        acc['gf']     += [tbl[t]['gf']  for t in teams]
        acc['ga']     += [tbl[t]['ga']  for t in teams]
        acc['xgf']    += [tbl[t]['xgf'] for t in teams]
        acc['xga']    += [tbl[t]['xga'] for t in teams]
        acc['pos']    += positions
        cnt['title']  += (positions == 1)
        cnt['top5']   += (positions <= 5)
        cnt['top8']   += (positions <= 8)
        cnt['rel']    += (positions >= 18)
        pos_freq[np.arange(n), positions - 1] += 1

    N = n_simulations
    var_pts = np.maximum(acc['pts_sq'] / N - (acc['pts'] / N) ** 2, 0)
    std_pts = np.sqrt(var_pts)

    avg_df = pd.DataFrame({
        'team':             teams,
        'avg_points':       acc['pts']  / N,
        'pts_low':          acc['pts']  / N - 1.28 * std_pts,
        'pts_high':         acc['pts']  / N + 1.28 * std_pts,
        'avg_wins':         acc['w']    / N,
        'avg_draws':        acc['d']    / N,
        'avg_losses':       acc['l']    / N,
        'avg_goals_for':    acc['gf']   / N,
        'avg_goals_against':acc['ga']   / N,
        'avg_xg_for':       acc['xgf']  / N,
        'avg_xg_against':   acc['xga']  / N,
        'avg_position':     acc['pos']  / N,
        'title_pct':        np.round(cnt['title'] / N * 100, 1),
        'top5_pct':         np.round(cnt['top5']  / N * 100, 1),
        'top8_pct':         np.round(cnt['top8']  / N * 100, 1),
        'relegation_pct':   np.round(cnt['rel']   / N * 100, 1),
    })
    avg_df['avg_goal_difference'] = avg_df['avg_goals_for']  - avg_df['avg_goals_against']
    avg_df['avg_xg_difference']   = avg_df['avg_xg_for']     - avg_df['avg_xg_against']
    avg_df = avg_df.sort_values(
        ['avg_points', 'avg_goal_difference', 'avg_goals_for'],
        ascending=False
    ).reset_index(drop=True)
    avg_df.index += 1

    position_freq = {t: list(pos_freq[i]) for i, t in enumerate(teams)}
    return avg_df, position_freq


# ── DB helpers ────────────────────────────────────────────────────────────────

def load_actual_results(db_path, league, season):
    """Load played match results from the `matches` table (real scores)."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            match_date,
            home.team_name AS home_team,
            fmd.home_goals,
            away.team_name AS away_team,
            fmd.away_goals
        FROM matches fmd
            JOIN team_id_mapping home ON home.team_id = fmd.home_team
            JOIN team_id_mapping away ON away.team_id = fmd.away_team
        WHERE fmd.league_id = ? AND fmd.season = ?
    """, conn, params=[league, season])
    conn.close()
    df['is_actual'] = True
    return df


def get_actual_standings(actual_results_dict, teams):
    """Current league table derived from actual_results dict."""
    s = {t: {'pts':0,'w':0,'d':0,'l':0,'gf':0,'ga':0} for t in teams}
    for (ht, at), (hg, ag, *_) in actual_results_dict.items():
        s[ht]['gf'] += hg;  s[ht]['ga'] += ag
        s[at]['gf'] += ag;  s[at]['ga'] += hg
        if hg > ag:
            s[ht]['pts'] += 3; s[ht]['w'] += 1; s[at]['l'] += 1
        elif hg == ag:
            s[ht]['pts'] += 1; s[ht]['d'] += 1
            s[at]['pts'] += 1; s[at]['d'] += 1
        else:
            s[at]['pts'] += 3; s[at]['w'] += 1; s[ht]['l'] += 1
    df = (pd.DataFrame(s).T
            .reset_index().rename(columns={'index': 'team'}))
    df['gd'] = df['gf'] - df['ga']
    df = df.sort_values(['pts','gd','gf'], ascending=False).reset_index(drop=True)
    df['table_pos'] = df.index + 1
    return df


# ── Form rating (no model needed — derived from weighted scoreline data) ──────

def form_net_rating(weighted_df,
                    home_pen_rate=BASELINE_HOME_PENS,
                    away_pen_rate=BASELINE_AWAY_PENS):
    """
    Weighted expected-goals net rating from scoreline data.
    Does NOT require a fitted model — uses the blended scoreline distributions.
    """
    me = (weighted_df.groupby('match_id')
          .apply(lambda x: pd.Series({
              'exp_home_goals': (x['home_goals'] * x['weight']).sum() / x['weight'].sum(),
              'exp_away_goals': (x['away_goals'] * x['weight']).sum() / x['weight'].sum(),
              'match_weight':    x['weight'].sum(),
          })).reset_index())
    mm = weighted_df[['match_id','home_team','away_team']].drop_duplicates()
    me = me.merge(mm, on='match_id')

    pen_avg = (home_pen_rate + away_pen_rate) / 2

    home_s = me.groupby('home_team').apply(lambda x: pd.Series({
        'gf': (x['exp_home_goals'] * x['match_weight']).sum(),
        'ga': (x['exp_away_goals'] * x['match_weight']).sum(),
        'w':   x['match_weight'].sum(),
    }))
    away_s = me.groupby('away_team').apply(lambda x: pd.Series({
        'gf': (x['exp_away_goals'] * x['match_weight']).sum(),
        'ga': (x['exp_home_goals'] * x['match_weight']).sum(),
        'w':   x['match_weight'].sum(),
    }))

    ts = pd.DataFrame(index=home_s.index)
    ts['gf_avg']     = (home_s['gf'] + away_s['gf']) / (home_s['w'] + away_s['w']) + pen_avg
    ts['ga_avg']     = (home_s['ga'] + away_s['ga']) / (home_s['w'] + away_s['w']) + pen_avg
    ts['net_rating'] = ts['gf_avg'] - ts['ga_avg']
    return ts.sort_values('net_rating', ascending=False)


# ── Substack helpers ──────────────────────────────────────────────────────────

def get_last_result(df_actual, team):
    inv = df_actual[(df_actual['home_team'] == team) | (df_actual['away_team'] == team)].copy()
    if inv.empty:
        return 'No result'
    if 'match_date' in inv.columns:
        inv = inv.sort_values('match_date')
    last    = inv.iloc[-1]
    is_home = last['home_team'] == team
    hg, ag  = int(last['home_goals']), int(last['away_goals'])
    opp     = last['away_team'] if is_home else last['home_team']
    tg      = hg if is_home else ag
    og      = ag if is_home else hg
    outcome = 'Win' if tg > og else ('Draw' if tg == og else 'Loss')
    venue   = 'Home' if is_home else 'Away'
    return f'{hg}–{ag} ({venue} {outcome} vs {opp})'


def get_form_string(df_actual, team, n=5):
    inv = df_actual[(df_actual['home_team'] == team) | (df_actual['away_team'] == team)].copy()
    if inv.empty:
        return ''
    if 'match_date' in inv.columns:
        inv = inv.sort_values('match_date')
    results = []
    for _, row in inv.iterrows():
        is_home = row['home_team'] == team
        tg = int(row['home_goals'] if is_home else row['away_goals'])
        og = int(row['away_goals'] if is_home else row['home_goals'])
        results.append('W' if tg > og else ('D' if tg == og else 'L'))
    return ''.join(results[-n:])


def rank_arrow(current_rank, prev_rank):
    if prev_rank is None:   return '●'
    if current_rank < prev_rank: return '▲'
    if current_rank > prev_rank: return '▼'
    return '▬'
