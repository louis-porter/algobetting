"""
radar_chart.py — two-season radar chart for team performance comparison

Axis ranges are derived from league_df (all teams in the league) so each
position on a spoke represents context within the full league distribution.
Alternating white/light-gray concentric bands replace individual label
backgrounds, keeping the chart clean and readable.

Usage:
    from viz.radar_chart import make_radar

    ATTACK = [
        ('npxG\\nper game',      'npxG_for',          False, '{:.2f}'),
        ('Goals\\nper game',     'Goals_for',          False, '{:.2f}'),
        ('Goals\\nvs npxG',      'Goals_minus_npxG',   False, '{:+.2f}'),
        ('Shots\\nper game',     'Shots_for_pg',        False, '{:.1f}'),
        ('npxG\\nper shot',      'npxG_per_shot_for',   False, '{:.3f}'),
    ]

    make_radar(
        comp=comp,                        # DataFrame indexed by season string
        metrics=ATTACK,
        season_a='2024-2025',
        season_b='2025-2026',
        title="FCK Attack: 2024-25 vs 2025-26",
        subtitle="FC København  ·  Superligaen  ·  higher = better",
        league_df=league_stats,           # all-teams DataFrame for axis scaling
        save_path="fck_radar_attack.png",
    )

Metric tuple: (label, column, invert, fmt)
  - invert=True  → lower raw value scores better on the radar (e.g. goals conceded)
  - axis range is computed as [min, max] across all rows in league_df
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from viz.team_colors import get_colors
from viz.utils import multicolor_text

fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Regular.ttf')
fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Bold.ttf')
plt.rcParams['font.family'] = 'Roboto'

def _fmt2dp(fmt, val):
    """Format val with fmt but cap decimal places at 2."""
    s = fmt.format(val)
    # If the format produced more than 2dp, reformat at 2dp preserving sign prefix
    if '.' in s:
        integer_part, dec_part = s.split('.')
        if len(dec_part) > 2:
            prefix = '+' if '+' in fmt else ''
            s = f'{val:{prefix}.2f}'
    return s

_BAND_EDGES  = [i / 10 for i in range(11)]                           # 0.0, 0.1, …, 1.0
_BAND_COLORS = ['#ececec' if i % 2 == 0 else '#ffffff' for i in range(10)]  # alternating
_SPOKE_COLOR = '#cccccc'
_LABEL_C     = '#333333'


def make_radar(
    comp,
    metrics,
    season_a,
    season_b,
    title,
    subtitle,
    league_df,
    save_path=None,
    color_a=None,
    color_b=None,
    figsize=(7, 8),
    label_a=None,
    label_b=None,
    team_a=None,
    team_b=None,
):
    """
    Draw a polar radar comparing two seasons, scaled to the full league range.

    Parameters
    ----------
    comp : pd.DataFrame
        Indexed by season string; columns must include all metric columns.
    metrics : list of (label, col, invert, fmt)
    season_a, season_b : str
        Row labels in comp (e.g. '2024-2025', '2025-2026').
    title, subtitle : str
    league_df : pd.DataFrame
        All-team season averages. Used to set axis lo/hi per metric.
        Only needs the metric columns to be present.
    save_path : str or None
    color_a, color_b : hex colour strings for season_a and season_b.
    figsize : tuple
    """
    # Auto-resolve colours from team registry if not explicitly provided
    if color_a is None or color_b is None:
        _ca, _cb = get_colors(team_a or '', team_b or team_a or '')
        color_a = color_a or _ca
        color_b = color_b or _cb

    # ── Compute axis ranges from league distribution (with buffer) ────────────
    # Buffer keeps worst/best teams off the very centre/edge of the radar
    axis_ranges = {}
    for _, col, _, _ in metrics:
        lo   = league_df[col].min()
        hi   = league_df[col].max()
        pad  = (hi - lo) * 0.1
        axis_ranges[col] = (lo - pad, hi + pad)

    labels = [m[0] for m in metrics]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def norm(val, lo, hi, invert):
        n = float(np.clip((val - lo) / (hi - lo), 0, 1))
        return 1 - n if invert else n

    def season_vals(season):
        vals = [norm(comp.loc[season, col], *axis_ranges[col], inv)
                for _, col, inv, _ in metrics]
        return vals + [vals[0]]

    va = season_vals(season_a)
    vb = season_vals(season_b)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)

    # ── Alternating concentric bands (drawn before data) ──────────────────────
    theta_ring = np.linspace(0, 2 * np.pi, 360)
    for i, (r_lo, r_hi) in enumerate(zip(_BAND_EDGES[:-1], _BAND_EDGES[1:])):
        ax.fill_between(theta_ring, r_lo, r_hi,
                        color=_BAND_COLORS[i], zorder=0)

    # Outer boundary ring
    ax.plot(theta_ring, np.ones_like(theta_ring),
            color=_SPOKE_COLOR, linewidth=0.8, zorder=1)

    # ── Data polygons — opaque fills, no outline ──────────────────────────────
    for i, (v, col) in enumerate([(va, color_a), (vb, color_b)]):
        ax.fill(angles, v, color=col, alpha=0.55, zorder=2 + i)

    # ── Turn off default matplotlib grid/ticks (we drew our own) ──────────────
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks([])   # no default tick labels — drawn manually below

    # Spoke labels: parallel to their spoke, flipped on southern hemisphere
    R_LABEL = 1.1
    for label_text, angle in zip(labels, angles[:-1]):
        angle_deg = np.degrees(angle)
        rot = (-angle_deg) % 360
        if rot > 180:
            rot -= 360
        if rot > 90:
            rot -= 180
        elif rot < -90:
            rot += 180
        ax.text(angle, R_LABEL, label_text,
                ha='center', va='center',
                rotation=rot, rotation_mode='anchor',
                fontsize=9, color=_LABEL_C,
                clip_on=False, zorder=10, fontdict={'weight':'bold'})

    # ── Per-ring tick labels — rotated along their spoke, same flip rule ─────
    tick_rs = [i / 10 for i in range(1, 11)]  # 0.1 … 1.0

    for i, (_, col, inv, fmt) in enumerate(metrics):
        angle     = angles[i]
        angle_deg = np.degrees(angle)
        rot = (-angle_deg) % 360
        if rot > 180:
            rot -= 360
        if rot > 90:
            rot -= 180
        elif rot < -90:
            rot += 180

        lo, hi = axis_ranges[col]
        for r in tick_rs:
            raw   = lo + (1 - r) * (hi - lo) if inv else lo + r * (hi - lo)
            label = _fmt2dp(fmt, raw)
            ax.text(angle, r, label,
                    ha='center', va='center',
                    rotation=rot, rotation_mode='anchor',
                    fontsize=8, color='#666666', zorder=5)

    # ── Title: centred, coloured by team/season ───────────────────────────────
    sa = label_a or season_a
    sb = label_b or season_b

    # ── Line 1: coloured identifier (always auto-generated) ──────────────────
    same_team = (team_a and team_b and team_a == team_b) or (team_a and not team_b)
    if same_team and team_a:
        multicolor_text(fig, ax, [
            (sa,       color_a),
            ('  vs  ', '#888888'),
            (sb,       color_b),
        ], y=0.978, fontsize=12, fontweight='bold', ref='center')
    elif team_a and team_b:
        multicolor_text(fig, ax, [
            (team_a,   color_a),
            ('  vs  ', '#333333'),
            (team_b,   color_b),
        ], y=0.978, fontsize=12, fontweight='bold', ref='center')
    else:
        fig.text(0.5, 0.99, title, ha='center', va='top',
                 fontsize=12, fontweight='bold', color='#222222')

    # ── Line 2: chart type label ──────────────────────────────────────────────
    fig.text(0.5, 0.97, title, ha='center', va='top',
             fontsize=11, fontweight='bold', color='#333333')

    # ── Line 3: context (league · season · per 90) ────────────────────────────
    fig.text(0.5, 0.95, subtitle, ha='center', va='top',
             fontsize=9, color='#888888')

    fig.subplots_adjust(top=0.97, bottom=0.05)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.show()
