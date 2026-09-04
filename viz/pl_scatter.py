"""
pl_scatter.py — Attack vs Defence scatter with club logos as markers.

Axes are goals for/against per game against an average opponent (same units
as the Team Ratings leaderboard, e.g. outputs.ipynb's `ratings_df`) rather
than the model's raw relative att/def log-strength units -- directly
comparable to "how many goals does this team actually score/concede against
a typical side," not just "better/worse than the league average" in an
abstract log scale. Attack is the y-axis (up = more goals scored, natural
orientation); defence is the x-axis, inverted so fewer goals conceded (a
better defence) is still to the right -- keeps the "best teams are up and to
the right" reading intact after the axis swap.

Usage
-----
    from pl_scatter import render_scatter

    fig = render_scatter(
        ratings = ratings_df,   # DataFrame with columns: team, goals_for, goals_against
        season  = '2025-2026',
        league  = 'Premier League',
    )
    plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import requests
from PIL import Image
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

for _path in ['/Users/admin/Library/Fonts/Roboto-Regular.ttf',
              '/Users/admin/Library/Fonts/Roboto-Bold.ttf']:
    try:
        fm.fontManager.addfont(_path)
    except Exception:
        pass
plt.rcParams['font.family'] = 'Roboto'

from logos import TEAM_LOGOS

_LOGO_PX = 44
_HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'}


def _fetch_logo(team):
    url = TEAM_LOGOS.get(team)
    if not url:
        return None
    try:
        r   = requests.get(url, timeout=5, headers=_HEADERS)
        img = Image.open(BytesIO(r.content)).convert('RGBA')
        # Trim transparent border
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        scale = _LOGO_PX / max(img.width, img.height)
        nw = max(1, round(img.width  * scale))
        nh = max(1, round(img.height * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
        sq  = Image.new('RGBA', (_LOGO_PX, _LOGO_PX), (0, 0, 0, 0))
        sq.paste(img, ((_LOGO_PX - nw) // 2, (_LOGO_PX - nh) // 2), mask=img)
        return sq
    except Exception:
        return None


def render_scatter(ratings, season='', league='Premier League',
                   save_path=None, figsize=(11, 9)):
    """
    Parameters
    ----------
    ratings  : DataFrame with columns team, goals_for, goals_against (per game
               against an average opponent -- e.g. outputs.ipynb's `ratings_df`)
    season   : str
    league   : str
    save_path: optional PNG path
    figsize  : tuple

    Returns
    -------
    matplotlib Figure
    """
    # Pre-fetch all logos
    logo_cache = {team: _fetch_logo(team) for team in ratings['team']}

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.tick_params(left=False, bottom=False)
    ax.yaxis.grid(True, color='#eeeeee', linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, color='#eeeeee', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Reference lines at the league's own average, not zero -- these are real goals-per-game
    # counts now, not mean-centered relative-strength units.
    avg_goals_for     = ratings['goals_for'].mean()
    avg_goals_against = ratings['goals_against'].mean()
    ax.axhline(avg_goals_for, color='#cccccc', linewidth=1.0, zorder=1)
    ax.axvline(avg_goals_against, color='#cccccc', linewidth=1.0, zorder=1)

    # Plot invisible points first so matplotlib autoscales to the actual data range
    # (AnnotationBbox objects don't trigger autoscaling)
    ax.scatter(ratings['goals_against'], ratings['goals_for'], s=0, alpha=0, zorder=0)

    for _, row in ratings.iterrows():
        logo = logo_cache.get(row['team'])
        if logo:
            im = OffsetImage(logo, zoom=72 / 160)
            ab = AnnotationBbox(im, (row['goals_against'], row['goals_for']),
                                frameon=False, zorder=3)
            ax.add_artist(ab)
        else:
            # Fallback: dot + label if logo missing
            ax.scatter(row['goals_against'], row['goals_for'], s=40,
                       color='#aaaaaa', zorder=3)
            ax.annotate(row['team'], (row['goals_against'], row['goals_for']),
                        textcoords='offset points', xytext=(5, 3), fontsize=7,
                        color='#666666')

    ax.invert_xaxis()
    ax.set_ylabel('Goals for', fontsize=9, color='#666666', labelpad=8)
    ax.set_xlabel('Goals against', fontsize=9, color='#666666', labelpad=8)
    ax.tick_params(axis='both', labelsize=9, labelcolor='#666666')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    subtitle = 'vs. Average Opponent'
    ax.text(0, 1.06, f'Attack/Defence Ratings',
            transform=ax.transAxes,
            fontsize=13, fontweight='bold', color='#222222', va='bottom', ha='left')
    ax.text(0, 1.04, subtitle,
            transform=ax.transAxes,
            fontsize=9, color='#888888', va='bottom', ha='left')

    plt.tight_layout()
    fig.subplots_adjust(top=0.91)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')

    return fig
