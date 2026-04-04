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

fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Regular.ttf')
fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Bold.ttf')
plt.rcParams['font.family'] = 'Roboto'

_TICK_LEVELS = [0.25, 0.5, 0.75, 1.0]

# Alternating band colours from centre outward
_BAND_COLORS = ['#ececec', '#ffffff', '#ececec', '#ffffff']
_SPOKE_COLOR = '#cccccc'
_LABEL_C     = '#333333'   # axis spoke labels
_TICK_C      = '#888888'   # ring tick value labels — readable on either band


def make_radar(
    comp,
    metrics,
    season_a,
    season_b,
    title,
    subtitle,
    league_df,
    save_path=None,
    color_a='#457b9d',
    color_b='#e63946',
    figsize=(7, 8),
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
    # ── Compute axis ranges from league distribution ───────────────────────────
    axis_ranges = {}
    for _, col, _, _ in metrics:
        lo = league_df[col].min()
        hi = league_df[col].max()
        axis_ranges[col] = (lo, hi)

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
    ring_edges = [0] + _TICK_LEVELS
    for i, (r_lo, r_hi) in enumerate(zip(ring_edges[:-1], ring_edges[1:])):
        ax.fill_between(theta_ring, r_lo, r_hi,
                        color=_BAND_COLORS[i], zorder=0)

    # Spoke lines (drawn over bands, under data)
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], color=_SPOKE_COLOR, linewidth=0.8, zorder=1)

    # Outer boundary ring
    ax.plot(theta_ring, np.ones_like(theta_ring),
            color=_SPOKE_COLOR, linewidth=0.8, zorder=1)

    # ── Data polygons ─────────────────────────────────────────────────────────
    for v, col in [(va, color_a), (vb, color_b)]:
        ax.plot(angles, v, color=col, linewidth=2, zorder=3)
        ax.fill(angles, v, color=col, alpha=0.15, zorder=2)
        ax.scatter(angles, v, color=col, s=40, zorder=4)

    # ── Turn off default matplotlib grid/ticks (we drew our own) ──────────────
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=_LABEL_C)
    ax.tick_params(pad=12)

    # ── Spoke tick labels — value at each ring boundary, no background ─────────
    for i, (_, col, inv, fmt) in enumerate(metrics):
        angle = angles[i]
        lo, hi = axis_ranges[col]
        for r in _TICK_LEVELS:
            raw = lo + (1 - r) * (hi - lo) if inv else lo + r * (hi - lo)
            ax.text(
                angle, r, fmt.format(raw),
                ha='center', va='center',
                fontsize=6.5, color=_TICK_C,
                zorder=5,
            )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color=color_a, linewidth=2, marker='o', markersize=5, label=season_a),
        plt.Line2D([0], [0], color=color_b, linewidth=2, marker='o', markersize=5, label=season_b),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              frameon=False, fontsize=10, ncol=2)

    # ── Title / subtitle ──────────────────────────────────────────────────────
    fig.text(0.5, 0.97, title,    ha='center', va='top', fontsize=13, fontweight='bold', color='#222222')
    fig.text(0.5, 0.93, subtitle, ha='center', va='top', fontsize=9,  color='#888888')
    fig.subplots_adjust(top=0.82, bottom=0.1)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.show()
