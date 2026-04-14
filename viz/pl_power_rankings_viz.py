"""
pl_power_rankings_viz.py — matplotlib Power Rankings table visualization.

Usage
-----
    from pl_power_rankings_viz import render_power_rankings

    fig = render_power_rankings(
        form_merged     = form_merged,
        df_actual       = df_actual,
        team_logos      = team_logos,
        gw_label        = 'GW31',
        form_end        = '2026-04-06',
        get_form_string = get_form_string,
        save_path       = 'outputs/power_rankings.png',
    )
    plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import requests
from PIL import Image
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import date

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
_BADGE_ZOOM = 96 / 300   # matches pl_combined_table exactly
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rounded_rect(ax, x, y, w, h, radius, color, zorder=3, alpha=1.0):
    fancy = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        linewidth=0, facecolor=color, zorder=zorder, alpha=alpha,
    )
    ax.add_patch(fancy)


def _draw_triangle(ax, cx, cy, size, direction='up', color='white', zorder=6):
    """Draw a solid triangle (up or down) centered at (cx, cy)."""
    h = size
    w = size * 0.88
    if direction == 'up':
        pts = [[cx,        cy + h * 0.52],
               [cx - w/2,  cy - h * 0.48],
               [cx + w/2,  cy - h * 0.48]]
    else:
        pts = [[cx,        cy - h * 0.52],
               [cx - w/2,  cy + h * 0.48],
               [cx + w/2,  cy + h * 0.48]]
    tri = plt.Polygon(pts, closed=True, facecolor=color, edgecolor='none', zorder=zorder)
    ax.add_patch(tri)


def _change_colors(delta):
    """
    Return (pill_bg, text_color) for a rank change of `delta` places.
    delta > 0 = moved up, delta < 0 = moved down, 0 = unchanged.
    Intensity scales linearly from 1 (weak) to 5+ (strongest).
    """
    if delta == 0:
        return '#D1D5DB', '#374151'

    t = min(abs(delta), 5) / 5.0   # 0.2 → 1.0

    if delta > 0:
        # light green (#DCFCE7) → dark green (#14532D)
        r = 0.863 + (0.078 - 0.863) * t
        g = 0.988 + (0.325 - 0.988) * t
        b = 0.906 + (0.176 - 0.906) * t
    else:
        # light red (#FEE2E2) → dark red (#7F1D1D)
        r = 0.996 + (0.498 - 0.996) * t
        g = 0.886 + (0.114 - 0.886) * t
        b = 0.886 + (0.114 - 0.886) * t

    pill = mcolors.to_hex((max(0, r), max(0, g), max(0, b)))
    text = 'white' if t >= 0.45 else '#1F2937'
    return pill, text


# ── Main render ───────────────────────────────────────────────────────────────
def render_power_rankings(form_merged, df_actual, team_logos,
                          gw_label='', form_end='', get_form_string=None,
                          save_path=None):
    """
    Render a Power Rankings table as a matplotlib figure.

    Parameters
    ----------
    form_merged     : DataFrame  index=team, cols: net_rating / gf_avg / ga_avg /
                                 form_rank / prev_form_rank
    df_actual       : DataFrame  actual match results (for form dots)
    team_logos      : dict       team → badge URL
    gw_label        : str        shown in header, e.g. 'GW31'
    form_end        : str        date string for subtitle
    get_form_string : callable   get_form_string(df_actual, team, n) → 'WDLLW'
    save_path       : str | None  if given, saves here at _SAVE_DPI

    Returns
    -------
    matplotlib Figure
    """
    if get_form_string is None:
        def get_form_string(df, team, n=5):
            return ''

    teams = list(form_merged.index)
    n     = len(teams)

    # ── Pre-fetch badges ───────────────────────────────────────────────────────
    badge_cache = {}
    for team in teams:
        url = team_logos.get(team)
        if url and url not in badge_cache:
            badge_cache[url] = _fetch_badge(url)

    # ── Layout ────────────────────────────────────────────────────────────────
    FIG_W   = 8.5
    ROW_H   = 0.46
    PAD_TOP = 1.05
    PAD_BOT = 0.30
    FIG_H   = n * ROW_H + PAD_TOP + PAD_BOT

    X_RANK    = 0.38   # rank number center
    X_BADGE   = 0.82   # badge center
    X_NAME    = 1.22   # team name left edge

    X_DOTS_L  = 3.78   # first form dot center (shifted left)
    DOT_GAP   = 0.23   # → last (5th) dot at 3.78 + 4×0.23 = 4.70

    # Net-rating bar: trough from X_BAR_L to X_BAR_R, zero at X_BAR_C
    X_BAR_L   = 5.40
    HALF_W    = 0.65   # wider bar
    X_BAR_C   = X_BAR_L + HALF_W          # 6.05 — the zero line
    X_BAR_R   = X_BAR_L + HALF_W * 2      # 6.70

    # Change badge: pill centered at X_CHANGE — pushed right for clearance
    X_CHANGE  = 7.80
    PILL_W    = 0.68
    PILL_H_F  = 0.52   # fraction of ROW_H

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('#FAFAFA')
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis('off')

    # ── Title ─────────────────────────────────────────────────────────────────
    title = f'PL Power Rankings  -  {gw_label}' if gw_label else 'PL Power Rankings'
    ax.text(0.18, FIG_H - 0.20, title,
            fontsize=17, fontweight='bold', color='#111', va='top', ha='left')
    # ax.text(0.18, FIG_H - 0.45,
    #         f'Updated: {date.today().strftime("%-d %b %Y")}',
    #         fontsize=8.5, color='#888', va='top', ha='left')
    
    # ── Column headers ────────────────────────────────────────────────────────
    hdr_y  = FIG_H - PAD_TOP + ROW_H * 0.55
    hdr_kw = dict(fontsize=7.5, color='#888', va='center', fontweight='bold')

    ax.text(X_RANK,               hdr_y, 'RNK',          ha='center', **hdr_kw)
    ax.text(X_NAME,               hdr_y, 'TEAM',          ha='left',   **hdr_kw)
    ax.text(X_DOTS_L + DOT_GAP*2, hdr_y, 'FORM (LAST 5)', ha='center', **hdr_kw)
    ax.text(X_BAR_C,              hdr_y, 'FORM RATING', ha='center', **hdr_kw)
    ax.text(X_CHANGE,             hdr_y, 'CHANGE',        ha='center', **hdr_kw)

    ax.axhline(FIG_H - PAD_TOP + 0.02,
               xmin=0.02, xmax=0.98, color='#DDDDDD', linewidth=0.8)

    # ── Bar range ─────────────────────────────────────────────────────────────
    all_net = form_merged['net_rating'].values
    vmax    = max(float(np.abs(all_net).max()) * 1.1, 0.25)

    # ── Rows ──────────────────────────────────────────────────────────────────
    for i, team in enumerate(teams):
        rank = i + 1
        row  = form_merged.loc[team]
        net  = float(row['net_rating'])
        prev_val = row.get('prev_form_rank', None)
        prev = (int(prev_val)
                if prev_val is not None and not (isinstance(prev_val, float) and np.isnan(prev_val))
                else None)
        delta = (prev - rank) if prev is not None else None

        ry = FIG_H - PAD_TOP - rank * ROW_H   # row bottom
        rc = ry + ROW_H / 2                    # row center

        # Row background — plain alternating, no zone tints
        bg = '#F4F4F4' if rank % 2 == 0 else '#FAFAFA'
        ax.add_patch(mpatches.Rectangle(
            (0, ry), FIG_W, ROW_H,
            linewidth=0, facecolor=bg, zorder=1,
        ))

        # ── Rank ──────────────────────────────────────────────────────────────
        ax.text(X_RANK, rc, str(rank),
                ha='center', va='center',
                fontsize=12, fontweight='bold', color='#222', zorder=3)

        # ── Badge ─────────────────────────────────────────────────────────────
        url = team_logos.get(team)
        img = badge_cache.get(url) if url else None
        if img is not None:
            oim = OffsetImage(img, zoom=_BADGE_ZOOM)
            ax.add_artist(AnnotationBbox(oim, (X_BADGE, rc), frameon=False, zorder=3))

        # ── Team name ─────────────────────────────────────────────────────────
        ax.text(X_NAME, rc, team,
                ha='left', va='center',
                fontsize=11.5, fontweight='bold', color='#1A1A1A', zorder=3)

        # ── Form dots ─────────────────────────────────────────────────────────
        _DOT_COLOR = {'W': '#22C55E', 'D': '#F59E0B', 'L': '#EF4444'}
        form_str   = get_form_string(df_actual, team, n=5)
        for di, ch in enumerate(form_str[-5:]):
            dx  = X_DOTS_L + di * DOT_GAP
            col = _DOT_COLOR.get(ch, '#CCCCCC')
            ax.add_patch(plt.Circle((dx, rc), 0.093, color=col, zorder=3))
            ax.text(dx, rc, ch,
                    ha='center', va='center',
                    fontsize=5.5, fontweight='bold', color='white', zorder=4)

        # ── Net-rating bar ────────────────────────────────────────────────────
        bar_h = ROW_H * 0.38
        bar_y = rc - bar_h / 2

        # Trough
        _rounded_rect(ax, X_BAR_L, bar_y, HALF_W * 2, bar_h,
                      radius=0.04, color='#E0E0E0', zorder=2)

        # Fill
        fill_w   = min(HALF_W * abs(net) / vmax, HALF_W)
        fill_col = '#22C55E' if net >= 0 else '#EF4444'
        if net >= 0:
            _rounded_rect(ax, X_BAR_C, bar_y, fill_w, bar_h,
                          radius=0.04, color=fill_col, zorder=3)
        else:
            _rounded_rect(ax, X_BAR_C - fill_w, bar_y, fill_w, bar_h,
                          radius=0.04, color=fill_col, zorder=3)

        # Zero line
        ax.plot([X_BAR_C, X_BAR_C], [bar_y, bar_y + bar_h],
                color='#AAAAAA', linewidth=0.9, zorder=4)

        # Value label — inside the fill once it's wide enough, otherwise outside
        sign     = '+' if net > 0 else ''
        lbl      = f'{sign}{net:.2f}'
        INSIDE_T = 0.36   # fill_w threshold to flip label inside bar
        if fill_w >= INSIDE_T:
            # Inside the fill: anchor near the outer edge, white text
            if net >= 0:
                lbl_x, lbl_ha = X_BAR_C + fill_w - 0.02, 'right'
            else:
                lbl_x, lbl_ha = X_BAR_C - fill_w + 0.02, 'left'
            lbl_col = 'white'
        else:
            # Outside the fill: stay within the trough, dark text
            if net >= 0:
                lbl_x  = min(X_BAR_C + fill_w + 0.04, X_BAR_R - 0.02)
                lbl_ha = 'left'
            else:
                lbl_x  = max(X_BAR_C - fill_w - 0.04, X_BAR_L + 0.02)
                lbl_ha = 'right'
            lbl_col = '#333'
        ax.text(lbl_x, rc, lbl,
                ha=lbl_ha, va='center',
                fontsize=7.5, color=lbl_col, zorder=4)

        # ── Change badge ──────────────────────────────────────────────────────
        pill_h  = ROW_H * PILL_H_F
        pill_bg, txt_col = _change_colors(delta if delta is not None else 0)
        if delta is None:
            pill_bg, txt_col = '#D1D5DB', '#374151'

        _rounded_rect(ax,
                      X_CHANGE - PILL_W / 2, rc - pill_h / 2,
                      PILL_W, pill_h,
                      radius=pill_h * 0.38,
                      color=pill_bg, zorder=3)

        if delta is None or delta == 0:
            ax.text(X_CHANGE, rc, '—',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color=txt_col, zorder=5)
        else:
            direction = 'up' if delta > 0 else 'down'
            tri_x  = X_CHANGE - 0.14
            num_x  = X_CHANGE + 0.12
            _draw_triangle(ax, tri_x, rc, size=0.115,
                           direction=direction, color=txt_col, zorder=5)
            ax.text(num_x, rc, str(abs(delta)),
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', color=txt_col, zorder=5)

        # Row divider
        ax.axhline(ry, xmin=0.01, xmax=0.99,
                   color='#E0E0E0', linewidth=0.4, zorder=2)

    if save_path:
        fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'Saved → {save_path}')

    return fig
