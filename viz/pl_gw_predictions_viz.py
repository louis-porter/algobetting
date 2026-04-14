"""
pl_gw_predictions_viz.py — Gameweek fixture predictions table.

Usage
-----
    from viz.pl_gw_predictions_viz import render_gw_predictions

    fig = render_gw_predictions(
        gw_df      = next_gw_df,    # DataFrame: Home/Away/Home xG/Away xG/Home W%/Draw%/Away W%
        team_logos = team_logos,    # dict team → badge URL
        gw_label   = 'GW33',
        save_path  = 'outputs/gw33_predictions.png',
    )
    plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import requests
from PIL import Image
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import date

try:
    from viz.team_colors import TEAM_COLORS
except ImportError:
    from team_colors import TEAM_COLORS

# ── Fonts ──────────────────────────────────────────────────────────────────────
for _path in ['/Users/admin/Library/Fonts/Roboto-Regular.ttf',
              '/Users/admin/Library/Fonts/Roboto-Bold.ttf']:
    try:
        fm.fontManager.addfont(_path)
    except Exception:
        pass
plt.rcParams['font.family'] = 'Roboto'

# ── Badge fetching ─────────────────────────────────────────────────────────────
_BADGE_PX   = 64
_BADGE_ZOOM = 96 / 300
_SAVE_DPI   = 200
_HEADERS    = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}


def _trim_transparent(img):
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img


def _fetch_badge(url):
    try:
        r   = requests.get(url, timeout=5, headers=_HEADERS)
        img = Image.open(BytesIO(r.content)).convert('RGBA')
        img = _trim_transparent(img)
        scale = _BADGE_PX / max(img.width, img.height)
        nw = max(1, round(img.width  * scale))
        nh = max(1, round(img.height * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
        sq  = Image.new('RGBA', (_BADGE_PX, _BADGE_PX), (0, 0, 0, 0))
        sq.paste(img, ((_BADGE_PX - nw) // 2, (_BADGE_PX - nh) // 2), mask=img)
        return sq
    except Exception:
        return None


def _rounded_rect(ax, x, y, w, h, radius, color, zorder=3, alpha=1.0):
    fancy = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        linewidth=0, facecolor=color, zorder=zorder, alpha=alpha,
    )
    ax.add_patch(fancy)


def _team_color(team):
    entry = TEAM_COLORS.get(team)
    return entry[0] if entry else '#888888'


def render_gw_predictions(gw_df, team_logos, gw_label='', save_path=None):
    """
    Render a gameweek predictions table as a matplotlib figure.

    Parameters
    ----------
    gw_df      : DataFrame  cols: Home / Away / Home xG / Away xG / Home W% / Draw% / Away W%
    team_logos : dict       team → badge URL
    gw_label   : str        e.g. 'GW33'
    save_path  : str | None

    Returns
    -------
    matplotlib Figure
    """
    fixtures = gw_df.reset_index(drop=True)
    n = len(fixtures)

    # ── Pre-fetch badges ───────────────────────────────────────────────────────
    all_teams = set(fixtures['Home']) | set(fixtures['Away'])
    badge_cache = {}
    for team in all_teams:
        url = team_logos.get(team)
        if url and url not in badge_cache:
            badge_cache[url] = _fetch_badge(url)

    # ── Layout ────────────────────────────────────────────────────────────────
    FIG_W   = 8.6
    ROW_H   = 0.54
    PAD_TOP = 0.95
    PAD_BOT = 0.30
    FIG_H   = n * ROW_H + PAD_TOP + PAD_BOT

    # X positions — figure centre = 4.25
    X_HOME_BADGE  = 0.38
    X_HOME_NAME_L = 0.70   # left-align home name right after badge
    X_HOME_XG     = 2.55   # home xG label centre
    BAR_L         = 2.85
    BAR_R         = 5.65
    BAR_W         = BAR_R - BAR_L   # 2.80 — wide bar
    X_AWAY_XG     = 5.95   # away xG label centre
    X_AWAY_NAME_R = 7.80   # right-align away name right before badge
    X_AWAY_BADGE  = 8.10

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('#FAFAFA')
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis('off')

    # ── Title ─────────────────────────────────────────────────────────────────
    title = f'PL Predictions  -  {gw_label}' if gw_label else 'PL Predictions'
    ax.text(0.18, FIG_H - 0.15, title,
            fontsize=17, fontweight='bold', color='#111', va='top', ha='left')
    # ax.text(0.18, FIG_H - 0.45,
    #         f'Updated: {date.today().strftime("%-d %b %Y")}',
    #         fontsize=8.5, color='#888', va='top', ha='left')

    # ── Column headers ─────────────────────────────────────────────────────────
    hdr_y  = FIG_H - PAD_TOP + ROW_H * 0.50
    hdr_kw = dict(fontsize=7.5, color='#888', va='center', fontweight='bold')

    ax.text(X_HOME_NAME_L,           hdr_y, 'HOME',           ha='left',   **hdr_kw)
    ax.text(X_HOME_XG,               hdr_y, 'PREDICTED\nGOALS',             ha='center', **hdr_kw)
    ax.text((BAR_L + BAR_R) / 2,    hdr_y, 'WIN PROBABILITY', ha='center', **hdr_kw)
    ax.text(X_AWAY_XG,               hdr_y, 'PREDICTED\nGOALS',             ha='center', **hdr_kw)
    ax.text(X_AWAY_NAME_R,           hdr_y, 'AWAY',           ha='right',  **hdr_kw)

    ax.axhline(FIG_H - PAD_TOP + 0.02,
               xmin=0.02, xmax=0.98, color='#DDDDDD', linewidth=0.8)

    # ── Rows ──────────────────────────────────────────────────────────────────
    for i, fx in fixtures.iterrows():
        home     = fx['Home']
        away     = fx['Away']
        home_xg  = fx['Home xG']
        away_xg  = fx['Away xG']
        home_win = fx['Home W%']
        draw_pct = fx['Draw%']
        away_win = fx['Away W%']

        rank = i + 1
        ry = FIG_H - PAD_TOP - rank * ROW_H
        rc = ry + ROW_H / 2

        # Row background
        bg = '#F4F4F4' if rank % 2 == 0 else '#FAFAFA'
        ax.add_patch(mpatches.Rectangle(
            (0, ry), FIG_W, ROW_H,
            linewidth=0, facecolor=bg, zorder=1,
        ))

        # ── Home badge ────────────────────────────────────────────────────────
        url = team_logos.get(home)
        img = badge_cache.get(url) if url else None
        if img is not None:
            oim = OffsetImage(img, zoom=_BADGE_ZOOM)
            ax.add_artist(AnnotationBbox(oim, (X_HOME_BADGE, rc), frameon=False, zorder=3))

        # ── Home name (left-aligned after badge) ─────────────────────────────
        ax.text(X_HOME_NAME_L, rc, home,
                ha='left', va='center',
                fontsize=11, color='#1A1A1A', zorder=3)

        # ── Home xG ───────────────────────────────────────────────────────────
        ax.text(X_HOME_XG, rc, f'{home_xg:.2f}',
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='#1A1A1A', zorder=3)

        # ── Stacked probability bar ───────────────────────────────────────────
        BAR_H  = ROW_H * 0.52
        bar_y  = rc - BAR_H / 2

        home_col = _team_color(home)
        away_col = _team_color(away)

        home_w = BAR_W * home_win / 100
        draw_w = BAR_W * draw_pct / 100
        away_w = BAR_W * away_win / 100

        def _rect(x, y, w, h, color, zorder):
            ax.add_patch(mpatches.Rectangle(
                (x, y), w, h, linewidth=0, facecolor=color, zorder=zorder,
            ))

        # Background trough
        _rect(BAR_L, bar_y, BAR_W, BAR_H, '#E0E0E0', 2)

        # Home segment (left)
        if home_w > 0.01:
            _rect(BAR_L, bar_y, home_w, BAR_H, home_col, 3)

        # Draw segment (middle)
        if draw_w > 0.01:
            _rect(BAR_L + home_w, bar_y, draw_w, BAR_H, '#9CA3AF', 3)

        # Away segment (right)
        if away_w > 0.01:
            _rect(BAR_R - away_w, bar_y, away_w, BAR_H, away_col, 3)

        # Segment dividers
        for div_x in [BAR_L + home_w, BAR_L + home_w + draw_w]:
            ax.plot([div_x, div_x], [bar_y, bar_y + BAR_H],
                    color='white', linewidth=1.2, zorder=4)

        # Percentage labels inside segments
        MIN_W_FOR_LABEL = 0.28
        label_kw = dict(va='center', fontsize=8, fontweight='bold', zorder=5)

        if home_w >= MIN_W_FOR_LABEL:
            ax.text(BAR_L + home_w / 2, rc, f'{home_win:.0f}%',
                    ha='center', color='white', **label_kw)

        if draw_w >= MIN_W_FOR_LABEL:
            ax.text(BAR_L + home_w + draw_w / 2, rc, f'{draw_pct:.0f}%',
                    ha='center', color='white', **label_kw)

        if away_w >= MIN_W_FOR_LABEL:
            ax.text(BAR_R - away_w / 2, rc, f'{away_win:.0f}%',
                    ha='center', color='white', **label_kw)

        # ── Away xG ───────────────────────────────────────────────────────────
        ax.text(X_AWAY_XG, rc, f'{away_xg:.2f}',
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='#1A1A1A', zorder=3)

        # ── Away name (right-aligned before badge) ───────────────────────────
        ax.text(X_AWAY_NAME_R, rc, away,
                ha='right', va='center',
                fontsize=11, color='#1A1A1A', zorder=3)

        # ── Away badge ────────────────────────────────────────────────────────
        url = team_logos.get(away)
        img = badge_cache.get(url) if url else None
        if img is not None:
            oim = OffsetImage(img, zoom=_BADGE_ZOOM)
            ax.add_artist(AnnotationBbox(oim, (X_AWAY_BADGE, rc), frameon=False, zorder=3))


        # Row divider
        ax.axhline(ry, xmin=0.01, xmax=0.99,
                   color='#E0E0E0', linewidth=0.4, zorder=2)

    # ── Footer ────────────────────────────────────────────────────────────────
    if save_path:
        fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'Saved → {save_path}')

    return fig
