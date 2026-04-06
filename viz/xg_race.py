"""
xg_race.py — StatsBomb-style cumulative xG race chart for a single match

Usage:
    from viz.xg_race import plot_xg_race

    plot_xg_race(
        db_path  = 'infra/data/db/fotmob.db',
        match_id = 5222322,
        save_path= 'fck_xg_race.png',
    )

Note: own goals may not appear in the FotMob shotmap and therefore won't
be reflected in the xG lines or scoreline markers.
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from viz.team_colors import get_colors
from viz.utils import multicolor_text

fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Regular.ttf')
fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Bold.ttf')
plt.rcParams['font.family'] = 'Roboto'

N_SIM = 10_000


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
    conn.close()
    if match.empty:
        raise ValueError(f"match_id {match_id} not found")
    return shots, match.iloc[0]


def _cumulative_series(shots_team, max_min):
    shots_team = shots_team.copy()
    shots_team['expectedGoals'] = pd.to_numeric(
        shots_team['expectedGoals'], errors='coerce').fillna(0)
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
    """Running scoreline from shot events. Own goals attributed to opposite side."""
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




def plot_xg_race(
    db_path,
    match_id,
    save_path=None,
    color_home=None,
    color_away=None,
    figsize=(12, 6),
):
    shots, info = _load_match(db_path, match_id)

    home_name = info['home_name'] or str(info['home_team'])
    away_name = info['away_name'] or str(info['away_team'])

    # Auto-resolve team colours (clash detection built into get_colors)
    if color_home is None or color_away is None:
        _ch, _ca   = get_colors(home_name, away_name)
        color_home = color_home or _ch
        color_away = color_away or _ca
    home_g    = int(info['home_goals'])
    away_g    = int(info['away_goals'])
    date_str  = pd.to_datetime(info['match_date']).strftime('%d %b %Y')
    league    = str(info['league_id']).replace('_', ' ')

    shots_xg = shots[shots['eventType'] != 'OwnGoal'].copy()
    shots_xg['expectedGoals'] = pd.to_numeric(
        shots_xg['expectedGoals'], errors='coerce').fillna(0)

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

    def cumxg_at(side, minute):
        xs, ys = (hx, hy) if side == 'home' else (ax_, ay)
        return float(ys[max(0, np.searchsorted(xs, minute, side='right') - 1)])

    # ── Y-axis range: snapped to nearest 0.5 above the data, minimal headroom ─
    data_max = max(hy[-1], ay[-1])
    y_max    = np.ceil((data_max + 0.3) / 0.5) * 0.5   # e.g. 5.81 → 6.5

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f9f9f9')
    fig.subplots_adjust(top=0.87, bottom=0.09, left=0.07, right=0.88)

    ax.axvspan(0,  45,      color='white',   alpha=0.35, zorder=0)
    ax.axvspan(45, max_min, color='#eeeeee', alpha=0.35, zorder=0)
    ax.axvline(45, color='#bbbbbb', linewidth=1, linestyle='--', zorder=1)
    ax.text(45.5, y_max * 0.98, 'HT', fontsize=8, color='#999999', va='top')

    # ── Step lines + fills ───────────────────────────────────────────────────
    ax.step(hx,  hy, where='post', color=color_home, linewidth=2.5, zorder=3)
    ax.step(ax_, ay, where='post', color=color_away, linewidth=2.5, zorder=3)
    ax.fill_between(hx,  hy, step='post', color=color_home, alpha=0.12, zorder=2)
    ax.fill_between(ax_, ay, step='post', color=color_away, alpha=0.12, zorder=2)

    # ── Goal markers: clean filled circle with white border ──────────────────
    for (minute, side, hs, as_) in goal_seq:
        col  = color_home if side == 'home' else color_away
        y_pt = cumxg_at(side, minute)

        ax.scatter(minute, y_pt, s=30, color=col, zorder=6,
                   #edgecolors='white', linewidths=2
                   )
        ax.plot([minute, minute], [0, y_pt], color=col,
                linewidth=1, linestyle=':', alpha=0.4, zorder=4)

        # Scoreline: nudged right to avoid sitting on the vertical dotted line
        ax.text(minute-1, y_pt + y_max * 0.01, f'{hs}–{as_}',
                ha='left', va='bottom', fontsize=8,
                color=col, fontweight='bold', zorder=9)

    # ── Y ticks every 0.5, x ticks at standard minutes ───────────────────────
    ax.set_xlim(0, max_min + 5)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.5))
    ax.yaxis.grid(True, color='#dddddd', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(range(0, int(max_min) + 1, 10))
    ax.set_xlabel('Minute', fontsize=9, color='#666666', labelpad=6)
    ax.set_ylabel('Cumulative xG', fontsize=9, color='#666666', labelpad=6)
    ax.tick_params(labelsize=9, labelcolor='#666666', length=0)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#cccccc')

    # ── End-of-line xG labels, separated if too close ────────────────────────
    label_x = max_min + 0.8
    min_sep  = y_max * 0.07
    ly_h, ly_a = float(hy[-1]), float(ay[-1])
    if abs(ly_h - ly_a) < min_sep:
        mid   = (ly_h + ly_a) / 2
        sign  = 1 if ly_h >= ly_a else -1
        ly_h  = mid + sign * min_sep / 2
        ly_a  = mid - sign * min_sep / 2
    ax.text(label_x, ly_h, f'{hy[-1]:.2f}',
            color=color_home, fontsize=10, fontweight='bold', va='center', ha='left')
    ax.text(label_x, ly_a, f'{ay[-1]:.2f}',
            color=color_away, fontsize=10, fontweight='bold', va='center', ha='left')

    # ── Win probability — bar at very top, labels underneath ─────────────────
    # Inset: hugs the top of the axes (2% gap), tall enough for bar + labels
    ax_prob = ax.inset_axes([0.01, 0.84, 0.33, 0.14])
    ax_prob.set_xlim(0, 1)
    ax_prob.set_ylim(0, 1)
    ax_prob.axis('off')

    # Bar occupies the top ~40% of the inset
    bar_y, bar_h = 0.58, 0.38
    for val, left, col in [
        (p_home,            0,               color_home),
        (p_draw,            p_home,          '#aaaaaa'),
        (p_away,            p_home + p_draw, color_away),
    ]:
        ax_prob.barh(bar_y + bar_h / 2, val, left=left, height=bar_h,
                     color=col, zorder=3)

    # # % label inside bar only if segment is wide enough
    # for val, left, col in [
    #     (p_home,            0,               'white'),
    #     (p_draw,            p_home,          'white'),
    #     (p_away,            p_home + p_draw, 'white'),
    # ]:
    #     if val > 0.08:
    #         ax_prob.text(left + val / 2, bar_y + bar_h / 2, f'{val:.0%}',
    #                      ha='center', va='center', fontsize=8,
    #                      color=col, fontweight='bold', zorder=4)

    # Labels below bar: home left, draw centre, away right
    label_y = 0.52   # in transAxes, just below the bar bottom
    ax_prob.text(0.0, label_y, f'{home_name} Win {p_home:.0%}',
                 ha='left', va='top', fontsize=7.5,
                 color=color_home, fontweight='bold', transform=ax_prob.transAxes)
    ax_prob.text(0.5, label_y, f'Draw {p_draw:.0%}',
                 ha='center', va='top', fontsize=7.5,
                 color='#888888', fontweight='bold', transform=ax_prob.transAxes)
    ax_prob.text(1.0, label_y, f'{away_name} Win {p_away:.0%}',
                 ha='right', va='top', fontsize=7.5,
                 color=color_away, fontweight='bold', transform=ax_prob.transAxes)

    # ── Coloured title ────────────────────────────────────────────────────────
    multicolor_text(fig, ax, [
        (home_name,          color_home),
        (f'  {home_g}–{away_g}  ', '#222222'),
        (away_name,          color_away),
    ], y=0.90, fontsize=13, fontweight='bold')

    fig.text(0.07, 0.895,
             f'{date_str}  ·  {league}  ·  xG: {hy[-1]:.2f} – {ay[-1]:.2f}',
             ha='left', va='top', fontsize=9, color='#888888')

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.show()
