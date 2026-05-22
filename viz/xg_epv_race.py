"""
xg_epv_race.py — xG race + EPV momentum strip

Two-panel layout:
  Top:    Cumulative xG race (step lines, win probability, goals, red cards)
  Bottom: EPV momentum — net home minus away, Gaussian-smoothed

Usage:
    from viz.xg_epv_race import plot_xg_epv_race
    plot_xg_epv_race('infra/data/db/fotmob.db', match_id=4813657)

Note: EPV requires the WhoScored scraper to have been run for the match
(stored in match_events.EPV). Falls back to plain xG race if absent.
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.path as mpath

# Portrait card marker sized in points (2:3 ratio), used for red card icons
_CARD_MARKER = mpath.Path(
    np.array([[-0.33, -0.5], [0.33, -0.5], [0.33, 0.5], [-0.33, 0.5], [-0.33, -0.5]]),
    [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
     mpath.Path.LINETO, mpath.Path.CLOSEPOLY],
)
from scipy.ndimage import gaussian_filter1d

from viz.team_colors import get_colors
from viz.utils import multicolor_text

fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Regular.ttf')
fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Bold.ttf')
plt.rcParams['font.family'] = 'Roboto'

N_SIM = 10_000

_MODEL_TO_EVENT = {
    'Man City':     'Manchester City',
    'Man United':   'Manchester United',
    'Nottm Forest': 'Nottingham Forest',
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_match(db_path, match_id):
    conn = sqlite3.connect(db_path)
    shots = pd.read_sql(f"""
        SELECT min, teamId, side, eventType, expectedGoals, playerName
        FROM shots WHERE match_id = {match_id} ORDER BY min
    """, conn)
    match = pd.read_sql(f"""
        SELECT m.home_team, m.away_team, m.home_goals, m.away_goals,
               m.match_date, m.league_id,
               th.team_name AS home_name, ta.team_name AS away_name
        FROM matches m
        LEFT JOIN team_id_mapping th ON m.home_team = th.team_id
        LEFT JOIN team_id_mapping ta ON m.away_team = ta.team_id
        WHERE m.match_id = {match_id} LIMIT 1
    """, conn)
    red_cards = pd.read_sql(f"""
        SELECT side, time FROM red_cards WHERE match_id = {match_id} ORDER BY time
    """, conn)
    conn.close()
    if match.empty:
        raise ValueError(f"match_id {match_id} not found")
    return shots, match.iloc[0], red_cards


def _find_event_match_id(conn, home_name, away_name, match_date):
    event_home = _MODEL_TO_EVENT.get(home_name, home_name)
    event_away = _MODEL_TO_EVENT.get(away_name, away_name)
    date_str = str(match_date)[:10]
    row = conn.execute(
        "SELECT DISTINCT matchId FROM match_events "
        "WHERE homeTeam=? AND awayTeam=? AND startDate LIKE ? LIMIT 1",
        (event_home, event_away, f"{date_str}%"),
    ).fetchone()
    return row[0] if row else None


def _load_epv_momentum(conn, event_match_id, max_min, sigma_min=3.0, dt=0.5):
    df = pd.read_sql(f"""
        SELECT minute, second, h_a, EPV
        FROM match_events
        WHERE matchId = {event_match_id} AND EPV IS NOT NULL
        ORDER BY minute, second
    """, conn)
    if df.empty:
        return None, None

    df['t'] = df['minute'] + df['second'].fillna(0) / 60
    bins  = np.arange(0, max_min + dt, dt)
    t_mid = bins[:-1] + dt / 2
    h_epv = np.zeros(len(t_mid))
    a_epv = np.zeros(len(t_mid))
    for i, (t0, t1) in enumerate(zip(bins[:-1], bins[1:])):
        m = (df['t'] >= t0) & (df['t'] < t1)
        h_epv[i] = df.loc[m & (df['h_a'] == 'h'), 'EPV'].sum()
        a_epv[i] = df.loc[m & (df['h_a'] == 'a'), 'EPV'].sum()

    net = gaussian_filter1d(h_epv - a_epv, sigma_min / dt)
    return t_mid, net



# ── xG helpers ────────────────────────────────────────────────────────────────

def _cumulative_series(shots_team, max_min):
    shots_team = shots_team.copy()
    shots_team['expectedGoals'] = (
        pd.to_numeric(shots_team['expectedGoals'], errors='coerce').fillna(0)
    )
    mins, cumxg, total = [0], [0], 0.0
    for _, row in shots_team.iterrows():
        total += row['expectedGoals']
        mins.append(row['min'])
        cumxg.append(total)
    mins.append(max_min)
    cumxg.append(total)
    return np.array(mins), np.array(cumxg)


def _bernoulli_probs(home_xgs, away_xgs):
    rng = np.random.default_rng(42)
    h = (rng.binomial(1, np.clip(home_xgs, 0, 1), (N_SIM, len(home_xgs))).sum(axis=1)
         if len(home_xgs) else np.zeros(N_SIM))
    a = (rng.binomial(1, np.clip(away_xgs, 0, 1), (N_SIM, len(away_xgs))).sum(axis=1)
         if len(away_xgs) else np.zeros(N_SIM))
    return (h > a).mean(), (h == a).mean(), (h < a).mean()


def _goal_sequence(shots):
    goals = shots[shots['eventType'].isin(['Goal', 'OwnGoal'])].sort_values('min')
    h, a, seq = 0, 0, []
    for _, g in goals.iterrows():
        side = ('away' if g['side'] == 'home' else 'home') if g['eventType'] == 'OwnGoal' else g['side']
        if side == 'home':
            h += 1
        else:
            a += 1
        seq.append((int(g['min']), side, h, a))
    return seq


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_xg_epv_race(
    db_path,
    match_id,
    save_path=None,
    color_home=None,
    color_away=None,
    figsize=(12, 8),
    epv_sigma=3.0,
):
    shots, info, red_cards = _load_match(db_path, match_id)
    home_name = info['home_name'] or str(info['home_team'])
    away_name = info['away_name'] or str(info['away_team'])

    if color_home is None or color_away is None:
        _ch, _ca   = get_colors(home_name, away_name)
        color_home = color_home or _ch
        color_away = color_away or _ca

    home_g   = int(info['home_goals'])
    away_g   = int(info['away_goals'])
    date_str = pd.to_datetime(info['match_date']).strftime('%d %b %Y')

    shots_xg = shots[shots['eventType'] != 'OwnGoal'].copy()
    shots_xg['expectedGoals'] = (
        pd.to_numeric(shots_xg['expectedGoals'], errors='coerce').fillna(0)
    )
    home_shots = shots_xg[shots_xg['side'] == 'home']
    away_shots = shots_xg[shots_xg['side'] == 'away']
    max_min = max(shots['min'].max() if not shots.empty else 90, 90) + 1

    hx, hy  = _cumulative_series(home_shots, max_min)
    ax_, ay = _cumulative_series(away_shots, max_min)
    p_home, p_draw, p_away = _bernoulli_probs(
        home_shots['expectedGoals'].values,
        away_shots['expectedGoals'].values,
    )
    goal_seq = _goal_sequence(shots)

    # ── EPV momentum ──────────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    event_id = _find_event_match_id(conn, home_name, away_name, info['match_date'])
    t_mid = net_mom = None
    if event_id:
        t_mid, net_mom = _load_epv_momentum(
            conn, event_id, max_min, sigma_min=epv_sigma
        )
    conn.close()
    has_epv = t_mid is not None

    # ── Figure / layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')

    if has_epv:
        gs = gridspec.GridSpec(
            2, 1, height_ratios=[3.5, 1], hspace=0.06,
            top=0.87, bottom=0.09, left=0.07, right=0.88,
        )
        ax_xg  = fig.add_subplot(gs[0])
        ax_mom = fig.add_subplot(gs[1], sharex=ax_xg)
    else:
        gs = gridspec.GridSpec(1, 1, top=0.87, bottom=0.09, left=0.07, right=0.88)
        ax_xg  = fig.add_subplot(gs[0])
        ax_mom = None

    # ── xG race panel ─────────────────────────────────────────────────────────
    y_max = np.ceil((max(hy[-1], ay[-1]) + 0.3) / 0.5) * 0.5

    ax_xg.set_facecolor('#f9f9f9')
    ax_xg.axvspan(0,  45,      color='white',   alpha=0.35, zorder=0)
    ax_xg.axvspan(45, max_min, color='#eeeeee', alpha=0.35, zorder=0)
    ax_xg.axvline(45, color='#bbbbbb', linewidth=1, linestyle='--', zorder=1)
    ax_xg.text(45.5, y_max * 0.98, 'HT', fontsize=8, color='#999999', va='top')

    ax_xg.step(hx,  hy, where='post', color=color_home, linewidth=2.5, zorder=3)
    ax_xg.step(ax_, ay, where='post', color=color_away, linewidth=2.5, zorder=3)
    ax_xg.fill_between(hx,  hy, step='post', color=color_home, alpha=0.12, zorder=2)
    ax_xg.fill_between(ax_, ay, step='post', color=color_away, alpha=0.12, zorder=2)

    def cumxg_at(side, minute):
        xs, ys = (hx, hy) if side == 'home' else (ax_, ay)
        return float(ys[max(0, np.searchsorted(xs, minute, side='right') - 1)])

    # Goals
    for minute, side, hs, as_ in goal_seq:
        col  = color_home if side == 'home' else color_away
        y_pt = cumxg_at(side, minute)
        ax_xg.scatter(minute, y_pt, s=30, color=col, zorder=6)
        ax_xg.plot([minute, minute], [0, y_pt], color=col,
                   linewidth=1, linestyle=':', alpha=0.4, zorder=4)
        ax_xg.text(minute - 1, y_pt + y_max * 0.01, f'{hs}–{as_}',
                   ha='left', va='bottom', fontsize=8,
                   color=col, fontweight='bold', zorder=9)
        if ax_mom is not None:
            ax_mom.axvline(minute, color=col, linewidth=1, linestyle=':', alpha=0.4, zorder=4)

    # Red cards — card-shaped marker (fixed point size) + dotted dropout to xT
    for _, rc in red_cards.iterrows():
        minute = int(rc['time'])
        side   = rc['side']
        col    = color_home if side == 'home' else color_away
        y_line = cumxg_at(side, minute)
        ax_xg.plot([minute, minute], [0, y_line], color=col,
                   linewidth=1, linestyle=':', alpha=0.4, zorder=4)
        if ax_mom is not None:
            ax_mom.axvline(minute, color=col, linewidth=1, linestyle=':', alpha=0.4, zorder=4)
        ax_xg.scatter(minute, y_line, marker=_CARD_MARKER, s=60,
                      facecolors='#dd0000', edgecolors='white', linewidths=0.5, zorder=8)

    ax_xg.set_xlim(0, max_min + 5)
    ax_xg.set_ylim(0, y_max)
    ax_xg.set_yticks(np.arange(0, y_max + 0.01, 0.5))
    ax_xg.yaxis.grid(True, color='#dddddd', linewidth=0.7, zorder=0)
    ax_xg.set_axisbelow(True)
    ax_xg.set_ylabel('Cumulative xG', fontsize=9, color='#666666', labelpad=6)
    ax_xg.tick_params(labelsize=9, labelcolor='#666666', length=0)
    for sp in ['top', 'right']:
        ax_xg.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax_xg.spines[sp].set_color('#cccccc')

    if has_epv:
        ax_xg.spines['bottom'].set_visible(False)
        ax_xg.tick_params(bottom=False)
        plt.setp(ax_xg.get_xticklabels(), visible=False)

    # End-of-line xG labels
    label_x = max_min + 0.8
    min_sep  = y_max * 0.07
    ly_h, ly_a = float(hy[-1]), float(ay[-1])
    if abs(ly_h - ly_a) < min_sep:
        mid   = (ly_h + ly_a) / 2
        sign  = 1 if ly_h >= ly_a else -1
        ly_h  = mid + sign * min_sep / 2
        ly_a  = mid - sign * min_sep / 2
    ax_xg.text(label_x, ly_h, f'{hy[-1]:.2f}',
               color=color_home, fontsize=10, fontweight='bold', va='center', ha='left')
    ax_xg.text(label_x, ly_a, f'{ay[-1]:.2f}',
               color=color_away, fontsize=10, fontweight='bold', va='center', ha='left')

    # Win probability inset
    ax_prob = ax_xg.inset_axes([0.01, 0.84, 0.33, 0.14])
    ax_prob.set_xlim(0, 1)
    ax_prob.set_ylim(0, 1)
    ax_prob.axis('off')
    bar_y, bar_h = 0.58, 0.38
    for val, left, col in [
        (p_home,            0,               color_home),
        (p_draw,            p_home,          '#aaaaaa'),
        (p_away,            p_home + p_draw, color_away),
    ]:
        ax_prob.barh(bar_y + bar_h / 2, val, left=left, height=bar_h, color=col, zorder=3)
    label_y = 0.52
    ax_prob.text(0.0, label_y, f'{home_name} Win {p_home:.0%}',
                 ha='left', va='top', fontsize=7.5, color=color_home,
                 fontweight='bold', transform=ax_prob.transAxes)
    ax_prob.text(0.5, label_y, f'Draw {p_draw:.0%}',
                 ha='center', va='top', fontsize=7.5, color='#888888',
                 fontweight='bold', transform=ax_prob.transAxes)
    ax_prob.text(1.0, label_y, f'{away_name} Win {p_away:.0%}',
                 ha='right', va='top', fontsize=7.5, color=color_away,
                 fontweight='bold', transform=ax_prob.transAxes)

    # ── EPV momentum panel ────────────────────────────────────────────────────
    if has_epv:
        ax_mom.set_facecolor('#f9f9f9')
        ax_mom.axhline(0, color='#cccccc', linewidth=0.8, zorder=2)
        ax_mom.axvline(45, color='#bbbbbb', linewidth=1, linestyle='--', zorder=1)
        ax_mom.axvspan(0,  45,      color='white',   alpha=0.35, zorder=0)
        ax_mom.axvspan(45, max_min, color='#eeeeee', alpha=0.35, zorder=0)

        ax_mom.fill_between(t_mid, 0, net_mom,
                            where=net_mom >= 0, interpolate=True,
                            color=color_home, alpha=0.80, zorder=3)
        ax_mom.fill_between(t_mid, 0, net_mom,
                            where=net_mom <  0, interpolate=True,
                            color=color_away, alpha=0.80, zorder=3)

        epv_peak = max(np.abs(net_mom).max(), 1e-6) * 1.25
        ax_mom.set_ylim(-epv_peak, epv_peak)
        ax_mom.set_yticks([])
        ax_mom.set_xticks(range(0, int(max_min) + 1, 10))
        ax_mom.set_xlabel('Minute', fontsize=9, color='#666666', labelpad=6)
        ax_mom.tick_params(labelsize=9, labelcolor='#666666', length=0)
        for sp in ['top', 'right', 'left']:
            ax_mom.spines[sp].set_visible(False)
        ax_mom.spines['bottom'].set_color('#cccccc')
        ax_mom.set_ylabel('xT Momentum', fontsize=9, color='#666666', labelpad=6)
    else:
        ax_xg.set_xticks(range(0, int(max_min) + 1, 10))
        ax_xg.set_xlabel('Minute', fontsize=9, color='#666666', labelpad=6)

    # ── Title ─────────────────────────────────────────────────────────────────
    multicolor_text(fig, ax_xg, [
        (home_name,          color_home),
        (f'  {home_g}–{away_g}  ', '#222222'),
        (away_name,          color_away),
    ], y=0.90, fontsize=13, fontweight='bold')

    fig.text(0.07, 0.895,
             f'{date_str}  ·  xG: {hy[-1]:.2f} – {ay[-1]:.2f}',
             ha='left', va='top', fontsize=9, color='#888888')

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.show()
