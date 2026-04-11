"""
pl_evo_chart.py — Datawrapper-style ratings evolution chart.

Each team gets its primary colour from team_colors.py.
Lines are labelled at their final point; no legend.
Call once per metric to get a separate, taller figure for each.

Usage
-----
    from pl_evo_chart import render_evo_chart

    for metric, title in [('net', 'Net strength'), ('att', 'Attack'), ('def', 'Defence')]:
        fig = render_evo_chart(evo_df, metric=metric, season='2025-2026')
        plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

for _path in ['/Users/admin/Library/Fonts/Roboto-Regular.ttf',
              '/Users/admin/Library/Fonts/Roboto-Bold.ttf']:
    try:
        fm.fontManager.addfont(_path)
    except Exception:
        pass
plt.rcParams['font.family'] = 'Roboto'

from team_colors import TEAM_COLORS

_DEFAULT_COLOR = '#aaaaaa'
_MIN_LABEL_GAP = 0.018   # data units — reduce if too sparse, increase if still overlapping


def _stagger(values, min_gap=_MIN_LABEL_GAP):
    """Nudge label y-positions apart so they don't overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos   = [values[i] for i in order]
    for _ in range(300):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < min_gap:
                mid        = (pos[i] + pos[i - 1]) / 2
                pos[i - 1] = mid - min_gap / 2
                pos[i]     = mid + min_gap / 2
                moved      = True
        if not moved:
            break
    out = [0.0] * len(values)
    for rank_i, orig_i in enumerate(order):
        out[orig_i] = pos[rank_i]
    return out


_METRIC_LABELS = {
    'net': 'Net strength (att − def)',
    'att': 'Attack strength',
    'def': 'Defence strength',
}

_METRIC_TITLES = {
    'net': 'Net Strength Evolution',
    'att': 'Attack Strength Evolution',
    'def': 'Defence Strength Evolution',
}


def render_evo_chart(evo_df, metric='net', season='', league='Premier League',
                     save_path=None, figsize=(15, 10)):
    """
    Render a single evolution chart for one metric.

    Parameters
    ----------
    evo_df   : DataFrame with columns gw, team, att, def, net
    metric   : 'net', 'att', or 'def'
    season   : str e.g. '2025-2026'
    league   : str
    save_path: optional PNG save path
    figsize  : tuple — taller = more vertical space between labels

    Returns
    -------
    matplotlib Figure
    """
    teams   = sorted(evo_df['team'].unique())
    gw_vals = sorted(evo_df['gw'].unique())

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.tick_params(left=False, bottom=False)
    ax.yaxis.grid(True, color='#eeeeee', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color='#dddddd', linewidth=1.0, zorder=1)

    # Collect final values for label stagger
    team_list, last_vals = [], []
    for team in teams:
        t_df = evo_df[evo_df['team'] == team].sort_values('gw')
        if t_df.empty:
            continue
        team_list.append(team)
        last_vals.append(float(t_df[metric].values[-1]))

    staggered = _stagger(last_vals)

    for team, last_raw, last_y in zip(team_list, last_vals, staggered):
        t_df  = evo_df[evo_df['team'] == team].sort_values('gw')
        color = TEAM_COLORS.get(team, (_DEFAULT_COLOR,))[0]
        gws   = t_df['gw'].values
        vals  = t_df[metric].values

        ax.plot(gws, vals, color=color, linewidth=1.8, zorder=3, solid_capstyle='round')
        ax.scatter(gws, vals, color=color, s=22, zorder=4)

        last_gw = gws[-1]
        # Small tick connector if label was nudged
        if abs(last_raw - last_y) > _MIN_LABEL_GAP * 0.3:
            ax.plot([last_gw, last_gw + 0.2], [last_raw, last_y],
                    color=color, linewidth=0.7, alpha=0.45, zorder=2)

        ax.text(last_gw + 0.3, last_y, team,
                fontsize=7.5, color=color, va='center', ha='left', fontweight='bold')

    ax.set_ylabel(_METRIC_LABELS[metric], fontsize=9, color='#666666', labelpad=8)
    ax.tick_params(axis='y', labelsize=9, labelcolor='#666666')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    ax.set_xticks(gw_vals)
    ax.set_xticklabels(
        [f'GW{g}' for g in gw_vals],
        fontsize=8, color='#444444', rotation=45, ha='right',
    )

    if metric == 'def':
        ax.invert_yaxis()

    subtitle = f'{season}  ·  GW5 onwards  ·  BEST model' if season else 'GW5 onwards  ·  BEST model'
    ax.text(0, 1.06, f'{league} — {_METRIC_TITLES[metric]}',
            transform=ax.transAxes,
            fontsize=13, fontweight='bold', color='#222222', va='bottom', ha='left')
    ax.text(0, 1.01, subtitle,
            transform=ax.transAxes,
            fontsize=9, color='#888888', va='bottom', ha='left')

    plt.tight_layout()
    fig.subplots_adjust(top=0.91, right=0.80)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')

    return fig
