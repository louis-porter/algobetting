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
import statsmodels.api as sm
import statsmodels.formula.api as smf

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


# ── Attack/defense (Dixon-Coles-shaped) penalty model ───────────────────────────
#
# `compute_penalty_axis_multipliers` below is a two-sided replacement for
# `compute_penalty_multipliers` above: an own-team "draws penalties" axis AND an
# opponent "concedes penalties" axis, each fit from season-level xG-for/xG-against with
# decay-weighted empirical-Bayes shrinkage pooled across seasons (not a single
# in-progress season's posterior). See analysis/penalties/penalty_dixon_coles.ipynb for
# the full derivation: xG turned out to be a tighter proxy for penalty rate than either
# gross_xT or the model's own attack-strength posterior (which showed ~zero correlation
# with penalty rate -- team *quality* isn't what draws penalties, volume is), and the
# concession axis is the stronger of the two (~11% out-of-sample deviance reduction vs
# ~3% for attack). Walk-forward, the combined structure beats the flat home/away
# baseline in every fold with any history to fit on (pooled log-loss 0.3576 vs 0.3606).
#
# Fit once, offline, via that notebook's rho/k grid search (cell 21) -- not re-fit live
# on every call, same convention as QUALITY_SLOPE above.
RHO_ATT, K_ATT = 0.3, 1   # attack axis (xg_for_pg -> pens_for): fast decay, light shrinkage
RHO_DEF, K_DEF = 0.7, 4   # defense axis (xg_against_pg -> pens_against): more pooling


def _prev_season(season):
    """'2026-2027' -> '2025-2026'."""
    start = int(season.split('-')[0])
    return f'{start - 1}-{start}'


def _season_standings(db_path, league, season):
    """Final table (pts, then GD, then GF) from REAL match results (actual goals,
    including penalties) -- not np_matches, whose penalty-stripped goals can disagree
    with reality on close relegation/promotion battles (checked empirically: it did,
    for 2025-2026 Premier League). Returns a DataFrame indexed by team_id, best team
    first, empty if there's no data for this league/season yet."""
    conn = sqlite3.connect(db_path)
    try:
        m = pd.read_sql_query(
            "SELECT home_team, away_team, home_goals, away_goals FROM matches "
            "WHERE league_id = ? AND season = ? AND home_goals IS NOT NULL",
            conn, params=[league, season],
        )
    finally:
        conn.close()

    if m.empty:
        return pd.DataFrame(columns=['pts', 'gd', 'gf'])

    teams = pd.unique(m[['home_team', 'away_team']].to_numpy().ravel())
    rows = []
    for t in teams:
        home = m[m.home_team == t]
        away = m[m.away_team == t]
        pts = ((home.home_goals > home.away_goals).sum() * 3 + (home.home_goals == home.away_goals).sum()
               + (away.away_goals > away.home_goals).sum() * 3 + (away.away_goals == away.home_goals).sum())
        gf = home.home_goals.sum() + away.away_goals.sum()
        ga = home.away_goals.sum() + away.home_goals.sum()
        rows.append(dict(team_id=t, pts=pts, gd=gf - ga, gf=gf))
    return pd.DataFrame(rows).set_index('team_id').sort_values(['pts', 'gd', 'gf'], ascending=False)


def _promotion_swap_map(db_path, target_league, feeder_league, target_season, team_ids, history_team_ids):
    """{new_team_id: donor_team_id} for teams in `team_ids` with no history anywhere in
    the pooled seasons (`history_team_ids`) -- i.e. genuinely new to the window, not just
    a team on a short losing/winning streak.

    Pairing convention (same one already validated in penalty_prediction.ipynb's
    original swap-map, e.g. Fulham 2022-2023 seeded from Leeds' trailing PL form):
    rank promoted teams by HOW they went up (`feeder_league`'s previous-season table --
    top 2 by points keep their rank, whichever of the new teams ISN'T in that top 2 is
    the playoff winner, rank 3, regardless of its actual table position), rank
    `target_league`'s previous-season bottom 3 worst-to-best-of-the-relegated, and pair
    rank-for-rank (best promoted <-> least-bad relegated, worst <-> worst).

    Returns {} if there's nothing to seed, or if last season's standings aren't
    available yet (e.g. this league's very first tracked season).
    """
    new_teams = [t for t in team_ids if t not in history_team_ids]
    if not new_teams:
        return {}

    prev_season = _prev_season(target_season)
    feeder_standings = _season_standings(db_path, feeder_league, prev_season)
    target_standings = _season_standings(db_path, target_league, prev_season)
    if feeder_standings.empty or len(target_standings) < 3:
        return {}

    feeder_rank_of = {t: i for i, t in enumerate(feeder_standings.index, start=1)}
    # Rank 1/2 keep their table position if they're actually a top-2 finisher; anyone
    # else among the known-new teams (a playoff winner, wherever it finished) is rank 3.
    promo_rank = sorted(new_teams, key=lambda t: feeder_rank_of.get(t, 3) if feeder_rank_of.get(t, 99) <= 2 else 3)

    relegated = target_standings.index[-3:].tolist()  # [18th (best), 19th, 20th (worst)]
    return dict(zip(promo_rank, relegated))


def _build_xg_season_panel(db_path, league, min_games=25):
    """Team-season panel (games, pens_for, pens_against, xg_for, xg_against, + _pg rates)
    for every completed team-season in `league`. Generalizes
    penalty_dixon_coles.ipynb's `build_season_team`, minus the zone/gross_xT columns
    (not used by the Q4/Q5 model that actually beat baseline -- see the module docstring
    above) so this only needs `matches` + `match_stats`, already what
    `compute_penalty_multipliers` reads today.

    Returns (panel_df, season_idx) where season_idx maps season string -> integer index
    (chronological order), used for the day-decay-free, season-level shrinkage below.
    """
    conn = sqlite3.connect(db_path)
    try:
        matches = pd.read_sql_query(
            "SELECT match_id, home_team, away_team, season FROM matches "
            "WHERE league_id = ? AND home_goals IS NOT NULL", conn, params=[league])
        pens = pd.read_sql_query("SELECT match_id, home_pens, away_pens FROM penalties", conn)
        ms = pd.read_sql_query(
            "SELECT match_id, home_expected_goals, away_expected_goals FROM match_stats", conn)
    finally:
        conn.close()

    matches = matches.merge(pens, on='match_id', how='left')
    matches[['home_pens', 'away_pens']] = matches[['home_pens', 'away_pens']].fillna(0)
    ms['home_expected_goals'] = pd.to_numeric(ms['home_expected_goals'], errors='coerce')
    ms['away_expected_goals'] = pd.to_numeric(ms['away_expected_goals'], errors='coerce')
    matches = matches.merge(ms, on='match_id', how='left')

    home = matches[['season', 'home_team', 'away_team', 'home_pens',
                     'home_expected_goals', 'away_expected_goals']].rename(columns={
        'home_team': 'team_id', 'away_team': 'opp_id', 'home_pens': 'pens_for',
        'home_expected_goals': 'xg_for', 'away_expected_goals': 'xg_against'})
    away = matches[['season', 'away_team', 'home_team', 'away_pens',
                     'away_expected_goals', 'home_expected_goals']].rename(columns={
        'away_team': 'team_id', 'home_team': 'opp_id', 'away_pens': 'pens_for',
        'away_expected_goals': 'xg_for', 'home_expected_goals': 'xg_against'})
    panel = pd.concat([home, away], ignore_index=True)

    against = panel[['season', 'team_id', 'opp_id', 'pens_for']].rename(
        columns={'team_id': 't', 'opp_id': 'team_id', 'pens_for': 'pens_against'})
    pa = against.groupby(['season', 'team_id'])['pens_against'].sum().reset_index()

    st = panel.groupby(['season', 'team_id']).agg(
        games=('pens_for', 'size'), pens_for=('pens_for', 'sum'),
        xg_for=('xg_for', 'sum'), xg_against=('xg_against', 'sum'),
    ).reset_index()
    st = st.merge(pa, on=['season', 'team_id'], how='left')
    st = st[st.games >= min_games].reset_index(drop=True)

    for num in ['pens_for', 'pens_against', 'xg_for', 'xg_against']:
        st[f'{num}_pg'] = st[num] / st['games']

    seasons = sorted(st.season.unique())
    season_idx = {s: i for i, s in enumerate(seasons)}
    st['sidx'] = st.season.map(season_idx)
    return st, season_idx


def _shrunk_effect(team_hist, league_mean_by_sidx, target_sidx, rho, k):
    """Decay-weighted, sample-size-shrunk deviation from the league mean, for one
    team's trailing seasons (`team_hist`: rows of sidx/val, strictly before
    target_sidx). `rho` is the per-season decay factor, `k` the empirical-Bayes
    pseudo-count pulling toward 0 (i.e. toward the league mean) when there's little
    history. See penalty_dixon_coles.ipynb cell 21."""
    if len(team_hist) == 0:
        return 0.0
    w = rho ** (target_sidx - 1 - team_hist['sidx'])
    dev = team_hist['val'] - team_hist['sidx'].map(league_mean_by_sidx)
    n_eff = w.sum()
    weighted_dev = (w * dev).sum() / n_eff if n_eff > 0 else 0.0
    return n_eff / (n_eff + k) * weighted_dev


def _one_step_ahead_features(st, season_idx, train_seasons, valcol, rho, k):
    """Leave-one-season-out within `train_seasons`: each training season's shrunk
    effect is built only from seasons before it, so the axis coefficient itself is fit
    without lookahead."""
    rows = []
    for i in range(len(train_seasons)):
        hist_seasons = train_seasons[:i]
        tgt_season = train_seasons[i]
        tgt_sidx = season_idx[tgt_season]
        hist_sidxs = [season_idx[s] for s in hist_seasons]
        league_mean = st[st.sidx.isin(hist_sidxs)].groupby('sidx')[valcol].mean()
        for team in st[st.season == tgt_season].team_id.unique():
            hist = st[(st.team_id == team) & (st.sidx.isin(hist_sidxs))][
                ['sidx', valcol]].rename(columns={valcol: 'val'})
            rows.append({'team_id': team, 'season': tgt_season,
                         'eff': _shrunk_effect(hist, league_mean, tgt_sidx, rho, k)})
    return pd.DataFrame(rows)


def _fit_penalty_axis(st, season_idx, train_seasons, target_sidx, rho, k, valcol,
                       target_count_col, team_ids, swap_map):
    """Fit `target_count_col ~ shrunk_effect(valcol)` (Poisson, offset=log(games)) on
    `train_seasons`, then return each of `team_ids`'s fitted rate for `target_sidx`.
    A team with no own history anywhere in `train_seasons` is seeded from its
    `swap_map` donor's history instead of falling back to the league average."""
    train_feat = _one_step_ahead_features(st, season_idx, train_seasons, valcol, rho, k)
    train_df = st[st.season.isin(train_seasons)].merge(train_feat, on=['team_id', 'season'], how='left')
    train_df['eff'] = train_df['eff'].fillna(0.0)
    model = smf.glm(f'{target_count_col} ~ eff', data=train_df, offset=np.log(train_df['games']),
                     family=sm.families.Poisson()).fit()

    train_sidxs = [season_idx[s] for s in train_seasons]
    league_mean_full = st[st.sidx.isin(train_sidxs)].groupby('sidx')[valcol].mean()

    rates = {}
    for team in team_ids:
        hist = st[(st.team_id == team) & (st.sidx.isin(train_sidxs))][
            ['sidx', valcol]].rename(columns={valcol: 'val'})
        if len(hist) == 0 and team in swap_map:
            donor = swap_map[team]
            hist = st[(st.team_id == donor) & (st.sidx.isin(train_sidxs))][
                ['sidx', valcol]].rename(columns={valcol: 'val'})
        eff = _shrunk_effect(hist, league_mean_full, target_sidx, rho, k)
        rates[team] = float(np.exp(model.params['Intercept'] + model.params['eff'] * eff))
    avg_rate = float(np.exp(model.params['Intercept']))
    return rates, avg_rate


def compute_penalty_axis_multipliers(db_path, league, feeder_league, target_season, team_mapping):
    """Two-sided penalty multipliers for `target_season`: an attack multiplier (own
    team's pens-drawn rate, from trailing xG-for) and a defense multiplier (own team's
    pens-conceded rate, from trailing xG-against), each pooled across every completed
    season in the DB with decay-weighted empirical-Bayes shrinkage (RHO_ATT/K_ATT,
    RHO_DEF/K_DEF above) rather than a single season's point estimate.

    This is a two-sided replacement for `compute_penalty_multipliers` -- see that
    function's docstring for the single-sided GD-based approach this supersedes for
    `league`. `feeder_league` is the league promoted teams come up from (e.g.
    'Championship' for 'Premier_League') -- used only for `_promotion_swap_map`, for any
    team in `team_mapping` with no history in the pooled seasons.

    Both returned dicts are renormalized to mean 1 across `team_mapping` -- this only
    redistributes the league's already-calibrated flat baseline (get_penalty_baseline)
    by relative quality, it doesn't move the league average either way.

    Returns (attack_multipliers, defense_multipliers), both {team_name: float}, meant to
    be passed as simulation.py's `pen_multipliers` / `pen_defense_multipliers`.
    """
    # `team_mapping`'s VALUES are a 0..n-1 positional index for indexing posterior arrays
    # (see data_utils.prepare_model_data) -- NOT the database team_id. Only its KEYS (team
    # names) are meaningful here; resolve the real DB team_id ourselves from
    # team_id_mapping, matching what _build_xg_season_panel's `matches`-derived team_id
    # column actually uses.
    team_names = list(team_mapping.keys())
    conn = sqlite3.connect(db_path)
    try:
        tid_table = pd.read_sql_query("SELECT team_id, team_name FROM team_id_mapping", conn)
    finally:
        conn.close()
    name_to_id = dict(zip(tid_table.team_name, tid_table.team_id))
    team_ids = [name_to_id[t] for t in team_names]
    id_to_name = dict(zip(team_ids, team_names))

    st, season_idx = _build_xg_season_panel(db_path, league)
    train_seasons = [s for s in sorted(season_idx, key=season_idx.get) if s != target_season]
    target_sidx = season_idx.get(target_season, max(season_idx.values(), default=-1) + 1)

    history_team_ids = set(st.team_id.unique())
    swap_map = _promotion_swap_map(db_path, league, feeder_league, target_season,
                                    team_ids, history_team_ids)

    attack_rates, avg_attack = _fit_penalty_axis(
        st, season_idx, train_seasons, target_sidx, RHO_ATT, K_ATT,
        'xg_for_pg', 'pens_for', team_ids, swap_map)
    defense_rates, avg_defense = _fit_penalty_axis(
        st, season_idx, train_seasons, target_sidx, RHO_DEF, K_DEF,
        'xg_against_pg', 'pens_against', team_ids, swap_map)

    attack_mult = pd.Series({id_to_name[t]: attack_rates[t] / avg_attack for t in team_ids})
    defense_mult = pd.Series({id_to_name[t]: defense_rates[t] / avg_defense for t in team_ids})
    attack_mult = attack_mult / attack_mult.mean()
    defense_mult = defense_mult / defense_mult.mean()
    return attack_mult.to_dict(), defense_mult.to_dict()
