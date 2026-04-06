"""
spider_chart.py — true spider/web chart (polygon rings, not circular)

Usage:
    from viz.spider_chart import make_spider

    make_spider(
        db_path  = '/path/to/fotmob.db',
        league   = 'Superligaen',
        team_a   = 'FC København',
        season_a = '2024-2025',
        season_b = '2025-2026',
        chart    = 'attack',        # 'attack' or 'defence'
        # optional second team (defaults to same team, different season)
        team_b   = 'FC Midtjylland',
        save_path= 'fck_spider.png',
    )

Metric definitions live in infra/data/feature_engineering/team_stats.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from infra.data.feature_engineering.team_stats import (
    get_standard_stats, ATTACK_METRICS, DEFENCE_METRICS,
)
from viz.team_colors import get_colors
from viz.utils import multicolor_text

fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Regular.ttf')
fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Bold.ttf')
plt.rcParams['font.family'] = 'Roboto'

_N_RINGS     = 10
_RING_LEVELS = [i / _N_RINGS for i in range(1, _N_RINGS + 1)]
_BAND_COLORS = ['#ececec' if i % 2 == 0 else '#ffffff' for i in range(_N_RINGS)]
_SPOKE_COLOR = '#cccccc'
_LABEL_C     = '#333333'

def _short_season(s):
    """'2024-2025' → '24-25'"""
    parts = s.split('-')
    return f"{parts[0][2:]}-{parts[1][2:]}" if len(parts) == 2 else s


def _spoke_rotation(theta_deg):
    """Text rotation perpendicular to spoke, never upside-down."""
    rot = (-theta_deg) % 360
    if rot > 180:
        rot -= 360
    if rot > 90:
        rot -= 180
    elif rot < -90:
        rot += 180
    return rot


def _draw_spider(ax, metrics, vals_a, vals_b, league_df,
                 label_a, label_b, color_a, color_b):
    """Internal: draw one spider onto an existing Axes."""

    axis_ranges = {}
    for _, col, _, _ in metrics:
        lo  = league_df[col].min()
        hi  = league_df[col].max()
        pad = (hi - lo) * 0.15
        axis_ranges[col] = (lo - pad, hi + pad)

    N      = len(metrics)
    labels = [m[0] for m in metrics]
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    sx     = np.sin(thetas)
    sy     = np.cos(thetas)

    def norm(val, lo, hi, invert):
        n = float(np.clip((val - lo) / (hi - lo), 0, 1))
        return 1 - n if invert else n

    def to_xy(vals):
        rs = np.array([norm(vals[col], *axis_ranges[col], inv)
                       for _, col, inv, _ in metrics])
        return rs * sx, rs * sy

    xa, ya = to_xy(vals_a)
    xb, yb = to_xy(vals_b)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)

    # Polygon bands (outside-in)
    th_c  = np.append(thetas, thetas[0])
    sx_c  = np.sin(th_c)
    sy_c  = np.cos(th_c)
    for band_idx in range(_N_RINGS - 1, -1, -1):
        r = _RING_LEVELS[band_idx]
        ax.fill(r * sx_c, r * sy_c, color=_BAND_COLORS[band_idx], zorder=0)

    # Ring outlines
    for r in _RING_LEVELS:
        ax.plot(r * sx_c, r * sy_c, color=_SPOKE_COLOR,
                linewidth=0.9 if r == 1.0 else 0.4, zorder=1)

    # Spokes
    for sxi, syi in zip(sx, sy):
        ax.plot([0, sxi], [0, syi], color=_SPOKE_COLOR, linewidth=0.5, zorder=1)

    # Data polygons
    for j, (xs, ys, col) in enumerate([(xa, ya, color_a), (xb, yb, color_b)]):
        ax.fill(np.append(xs, xs[0]), np.append(ys, ys[0]),
                color=col, alpha=0.55, zorder=2 + j)
        ax.scatter(xs, ys, color=col, s=45, zorder=4 + j)

    # Data labels
    RADIAL, LATERAL = 0.14, 0.09
    px, py = -sy, sx   # perpendicular unit vectors

    for i, (_, col, _, fmt) in enumerate(metrics):
        raw_a = fmt.format(vals_a[col])
        raw_b = fmt.format(vals_b[col])
        ax.annotate(raw_a,
                    xy=(xa[i], ya[i]),
                    xytext=(xa[i] + RADIAL*sx[i] + LATERAL*px[i],
                            ya[i] + RADIAL*sy[i] + LATERAL*py[i]),
                    fontsize=8, color=color_a, fontweight='bold',
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color=color_a, lw=0.6, alpha=0.5),
                    zorder=6)
        ax.annotate(raw_b,
                    xy=(xb[i], yb[i]),
                    xytext=(xb[i] + RADIAL*sx[i] - LATERAL*px[i],
                            yb[i] + RADIAL*sy[i] - LATERAL*py[i]),
                    fontsize=8, color=color_b, fontweight='bold',
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color=color_b, lw=0.6, alpha=0.5),
                    zorder=6)

    # Axis titles
    R_LABEL = 1.22
    for i, (label_text, theta) in enumerate(zip(labels, thetas)):
        rot = _spoke_rotation(np.degrees(theta))
        ax.text(R_LABEL * sx[i], R_LABEL * sy[i], label_text,
                ha='center', va='center',
                rotation=rot, rotation_mode='anchor',
                fontsize=10, color=_LABEL_C, zorder=10)

    # No legend — team/season identification handled by coloured title in make_spider


def make_spider(
    db_path,
    league,
    team_a,
    season_a,
    season_b,
    chart='attack',
    team_b=None,
    title=None,
    subtitle=None,
    save_path=None,
    color_a=None,
    color_b=None,
    figsize=(7, 8),
):
    """
    Draw a spider chart for a team (or two teams) across one or two seasons.

    Parameters
    ----------
    db_path  : str   Path to fotmob.db
    league   : str   e.g. 'Superligaen'
    team_a   : str   Primary team name (exact match to team_id_mapping)
    season_a : str   First season, e.g. '2024-2025'
    season_b : str   Second season (or same season if comparing two teams)
    chart    : str   'attack' or 'defence'
    team_b   : str or None  If provided, compare team_a/season_a vs team_b/season_b
    title    : str or None  Auto-generated if omitted
    subtitle : str or None
    """
    metrics   = ATTACK_METRICS if chart == 'attack' else DEFENCE_METRICS
    team_b    = team_b or team_a

    # Auto-resolve colours from team registry if not explicitly provided
    if color_a is None or color_b is None:
        _ca, _cb = get_colors(team_a, team_b)
        color_a = color_a or _ca
        color_b = color_b or _cb

    seasons   = list({season_a, season_b})

    stats     = get_standard_stats(db_path, league, seasons=seasons)
    league_df = stats  # all teams used for axis scaling

    def _row(team, season):
        mask = (stats['team_name'] == team) & (stats['season'] == season)
        rows = stats[mask]
        if rows.empty:
            raise ValueError(f"No data for {team!r} in {season!r} ({league})")
        return rows.iloc[0]

    vals_a = _row(team_a, season_a)
    vals_b = _row(team_b, season_b)

    # Auto legend labels
    label_a = f"{team_a} {_short_season(season_a)}"
    label_b = f"{team_b} {_short_season(season_b)}"

    # Auto title
    if title is None:
        chart_label = chart.capitalize()
        if team_b == team_a:
            title = f"{team_a} — {chart_label}: {_short_season(season_a)} vs {_short_season(season_b)}"
        else:
            title = f"{team_a} vs {team_b} — {chart_label}: {_short_season(season_a)}"

    if subtitle is None:
        subtitle = f"{league}  ·  higher = better on all axes"

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    _draw_spider(ax, metrics, vals_a, vals_b, league_df,
                 label_a, label_b, color_a, color_b)

    # ── Title: coloured by team/season ────────────────────────────────────────
    if team_b == team_a:
        # Same team: plain title, coloured seasons underneath
        fig.text(0.5, 0.98, title, ha='center', va='top',
                 fontsize=13, fontweight='bold', color='#222222')
        multicolor_text(fig, ax, [
            (_short_season(season_a), color_a),
            ('  vs  ',               '#888888'),
            (_short_season(season_b), color_b),
        ], y=0.945, fontsize=10, fontweight='bold', ref='center')
        fig.text(0.5, 0.925, subtitle, ha='center', va='top',
                 fontsize=9, color='#888888')
    else:
        # Two different teams: team names in their colours
        multicolor_text(fig, ax, [
            (team_a,   color_a),
            ('  vs  ', '#222222'),
            (team_b,   color_b),
        ], y=0.98, fontsize=13, fontweight='bold', ref='center')
        fig.text(0.5, 0.945, subtitle, ha='center', va='top',
                 fontsize=9, color='#888888')

    fig.subplots_adjust(top=0.90, bottom=0.08)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.show()
