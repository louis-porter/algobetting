"""
pl_combined_table.py — 538-style combined ratings + projections + position distribution chart.

Usage
-----
    from pl_combined_table import render_combined_table

    fig = render_combined_table(
        avg_table      = avg_table,       # from run_multiple_seasons
        position_freq  = position_freq,   # from run_multiple_seasons
        ratings_df     = ratings_df,      # avg xGF/xGA vs all opponents
        n_sims         = 10_000,
        season         = '2024-2025',
        team_logos     = team_logos,      # dict of team → badge URL
    )
    plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import requests
from PIL import Image
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import date

# Load Roboto if available; fall back gracefully
for _path in ['/Users/admin/Library/Fonts/Roboto-Regular.ttf',
              '/Users/admin/Library/Fonts/Roboto-Bold.ttf']:
    try:
        fm.fontManager.addfont(_path)
    except Exception:
        pass
plt.rcParams['font.family'] = 'Roboto'

ZONES = [
    ('Title',     [1],                '#F5C518'),
    ('Top 5',     [2, 3, 4, 5],       '#2d9e5f'),
    ('Top 8',     [6, 7, 8],          '#60A5FA'),
    ('Mid-table', list(range(9, 18)), '#cccccc'),
    ('Relegated', [18, 19, 20],       '#c0392b'),
]
_POS_COLOR = {p: c for _, positions, c in ZONES for p in positions}

_SAVE_DPI        = 300
_BADGE_PX        = 64


def _trim_transparent(img):
    """Crop away transparent/white border so the crest fills the frame."""
    bbox = img.getbbox()   # bounding box of non-zero alpha pixels
    if bbox:
        img = img.crop(bbox)
    return img


_HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'}


def _fetch_badge(url):
    try:
        r   = requests.get(url, timeout=4, headers=_HEADERS)
        img = Image.open(BytesIO(r.content)).convert('RGBA')
        img = _trim_transparent(img)
        scale = _BADGE_PX / max(img.width, img.height)
        nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
        img    = img.resize((nw, nh), Image.LANCZOS)
        sq = Image.new('RGBA', (_BADGE_PX, _BADGE_PX), (0, 0, 0, 0))
        sq.paste(img, ((_BADGE_PX - nw) // 2, (_BADGE_PX - nh) // 2), mask=img)
        return sq
    except Exception:
        return None


def _xgd_color(val, vmin=-1.5, vmax=1.5):
    norm = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    if norm >= 0.5:
        t = (norm - 0.5) * 2
        return mcolors.to_hex((1 - t*0.8, 1 - t*0.3, 1 - t*0.6))
    else:
        t = (0.5 - norm) * 2
        return mcolors.to_hex((1 - t*0.2, 1 - t*0.8, 1 - t*0.8))


def _stat_font_color(val, avg, spread, good='high'):
    t = (val - avg) / spread
    if good == 'low':
        t = -t
    t = max(-1.0, min(1.0, t))
    if t >= 0:
        r, g, b = 0.47 - t*0.38, 0.47 + t*0.18, 0.47 - t*0.25
    else:
        s = -t
        r, g, b = 0.47 + s*0.28, 0.47 - s*0.35, 0.47 - s*0.35
    return mcolors.to_hex((max(0, r), max(0, g), max(0, b)))


def _zone_cell_color(zone_hex, intensity):
    r, g, b = mcolors.to_rgb(zone_hex)
    t = intensity ** 0.5
    return mcolors.to_hex((1 + (r-1)*t, 1 + (g-1)*t, 1 + (b-1)*t))


def render_combined_table(avg_table, position_freq, ratings_df,
                          n_sims, season, team_logos):
    """
    Render the combined 538-style table.

    Parameters
    ----------
    avg_table     : DataFrame from run_multiple_seasons
    position_freq : dict from run_multiple_seasons
    ratings_df    : DataFrame with columns team / goals_for / goals_against / goal_diff
    n_sims        : int
    season        : str  e.g. '2024-2025'
    team_logos    : dict  team_name → badge URL

    Returns
    -------
    matplotlib Figure
    """
    # Pre-fetch badges
    badge_cache = {}
    for url in set(team_logos.values()):
        if url:
            badge_cache[url] = _fetch_badge(url)

    teams_sorted   = avg_table.sort_values('avg_position')['team'].tolist()
    n_teams        = len(teams_sorted)
    ratings_lookup = ratings_df.set_index('team') if ratings_df is not None else None

    # ── Layout ───────────────────────────────────────────────────────────────
    ROW_H   = 0.46
    PAD_TOP = 1.20
    PAD_BOT = 0.30
    FIG_W   = 14.2
    FIG_H   = n_teams * ROW_H + PAD_TOP + PAD_BOT

    X_RANK  = 0.22
    X_BADGE = 0.54
    X_NAME  = 1.12

    # Team rating — xGD, xGF, xGA
    SEC1_L  = 3.10;  X_XGD = 3.62;  X_XGF = 4.42;  X_XGA = 5.22;  SEC1_R = 5.72

    # Season projections — PTS, GD, four pct columns
    SEC2_L  = 5.86
    X_PTS   = 6.30;  X_PGDD = 7.08
    X_TITLE = 7.88;  X_TOP5 = 8.66;  X_TOP8 = 9.44;  X_REL = 10.22
    SEC2_R  = 10.74

    # Position bars
    X_BAR_L = 10.88;  X_BAR_R = FIG_W - 0.20;  BAR_W = X_BAR_R - X_BAR_L

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('white')
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.axis('off')

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(X_RANK, FIG_H - 0.22, 'Premier League Projections',
            fontsize=19, fontweight='bold', color='#111', va='top', ha='left')
    ax.text(X_RANK, FIG_H - 0.55,
            f'BEST model  ·  {n_sims:,} simulations  ·  {season}',
            fontsize=9, color='#999', va='top', ha='left')
    today_str = date.today().strftime('%-d %b %Y')
    ax.text(FIG_W - 0.28, FIG_H - 0.28, f'Updated: {today_str}',
            fontsize=9, color='#111', va='top', ha='right')

    # ── Section header bars ───────────────────────────────────────────────────
    sec_bar_y = FIG_H - PAD_TOP + 0.46
    sec_bar_h = 0.18
    sec_lbl_y = sec_bar_y + sec_bar_h / 2
    for lx, rx, label in [
        (SEC1_L,  SEC1_R,  'TEAM RATING'),
        (SEC2_L,  SEC2_R,  'SEASON PROJECTIONS'),
        (X_BAR_L, X_BAR_R, 'POSITION'),
    ]:
        ax.add_patch(mpatches.Rectangle(
            (lx, sec_bar_y), rx - lx, sec_bar_h,
            facecolor='#111', edgecolor='none', zorder=1, alpha=0.85))
        ax.text((lx + rx) / 2, sec_lbl_y, label,
                fontsize=7.5, fontweight='bold', color='white',
                va='center', ha='center', zorder=2)

    # ── Column headers ────────────────────────────────────────────────────────
    hdr_y = FIG_H - PAD_TOP + 0.18
    for x, lbl in [
        (X_NAME,  'Team'),    (X_XGD, 'Ovr.'),  (X_XGF, 'Att.'),  (X_XGA, 'Def.'),
        (X_PTS,   'Pts'),     (X_PGDD, 'GD'),
        (X_TITLE, 'Title %'), (X_TOP5, 'Top 5 %'), (X_TOP8, 'Top 8 %'), (X_REL, 'Rel %'),
    ]:
        ax.text(x, hdr_y, lbl, fontsize=7.8, color='#555',
                va='center', ha='center')

    ax.plot([0.15, FIG_W-0.15], [hdr_y-0.14]*2, color='#ddd', lw=0.7)

    slot_w = BAR_W / 20
    for pos in [1, 5, 10, 15, 20]:
        x = X_BAR_L + (pos - 1) * slot_w + slot_w / 2
        ax.text(x, hdr_y, str(pos), fontsize=7.5, color='#999', va='center', ha='center')

    # ── Rows ──────────────────────────────────────────────────────────────────
    avg_xgf = ratings_df['goals_for'].mean()  if ratings_lookup is not None else 1.35
    avg_xga = ratings_df['goals_against'].mean() if ratings_lookup is not None else 1.35
    spread  = max(ratings_df['goals_for'].std(), 0.1) if ratings_lookup is not None else 0.45

    for i, team in enumerate(teams_sorted):
        row_top = FIG_H - PAD_TOP - i * ROW_H
        row_mid = row_top - ROW_H / 2

        ax.add_patch(mpatches.Rectangle(
            (0.15, row_top - ROW_H + 0.03), FIG_W - 0.30, ROW_H - 0.03,
            facecolor='#f8f8f8' if i % 2 == 0 else 'white',
            edgecolor='none', zorder=0))

        ax.text(X_RANK, row_mid, str(i + 1),
                fontsize=9, va='center', ha='center', color='#aaa')

        url = team_logos.get(team)
        if url and badge_cache.get(url) is not None:
            im = OffsetImage(badge_cache[url], zoom=96/_SAVE_DPI)
            ax.add_artist(AnnotationBbox(im, (X_BADGE, row_mid), frameon=False, zorder=2))

        ax.text(X_NAME, row_mid, team,
                fontsize=9.5, va='center', ha='left', color='#111', fontweight='bold')

        # Ratings
        if ratings_lookup is not None and team in ratings_lookup.index:
            r   = ratings_lookup.loc[team]
            xgd, xgf, xga = r['goal_diff'], r['goals_for'], r['goals_against']
            cw, ch = 0.65, ROW_H * 0.72
            ax.add_patch(mpatches.FancyBboxPatch(
                (X_XGD - cw/2, row_mid - ch/2), cw, ch,
                boxstyle='round,pad=0.02', facecolor=_xgd_color(xgd),
                edgecolor='none', zorder=1))
            ax.text(X_XGD, row_mid, f'{xgd:+.2f}',
                    fontsize=9, va='center', ha='center', color='#111', fontweight='bold')
            ax.text(X_XGF, row_mid, f'{xgf:.2f}', fontsize=9, va='center', ha='center',
                    color=_stat_font_color(xgf, avg_xgf, spread, 'high'), fontweight='bold')
            ax.text(X_XGA, row_mid, f'{xga:.2f}', fontsize=9, va='center', ha='center',
                    color=_stat_font_color(xga, avg_xga, spread, 'low'), fontweight='bold')

        # Projections
        rd = avg_table[avg_table['team'] == team]
        if not rd.empty:
            r = rd.iloc[0]
            ax.text(X_PTS, row_mid, f'{r["avg_points"]:.1f}',
                    fontsize=9.5, va='center', ha='center', color='#111', fontweight='bold')
            v = r.get('avg_goal_difference', float('nan'))
            if not np.isnan(v):
                ax.text(X_PGDD, row_mid, f'{v:+.0f}', fontsize=9, va='center', ha='center', color='#333')

            for x, key, zone_col in [
                (X_TITLE, 'title_pct',      '#F5C518'),
                (X_TOP5,  'top5_pct',       '#2d9e5f'),
                (X_TOP8,  'top8_pct',       '#60A5FA'),
                (X_REL,   'relegation_pct', '#c0392b'),
            ]:
                val = r.get(key, 0.0)
                if val > 0.4:
                    intensity = min(1.0, val / 60)
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (x - 0.37, row_mid - ROW_H*0.36), 0.74, ROW_H*0.72,
                        boxstyle='round,pad=0.02',
                        facecolor=_zone_cell_color(zone_col, intensity),
                        edgecolor='none', zorder=1))
                lbl = f'{val:.1f}%' if val >= 1 else ('<1%' if val > 0 else '—')
                ax.text(x, row_mid, lbl, fontsize=8.5, va='center', ha='center', color='#111')

        # Per-position bars
        counts   = position_freq.get(team, [0]*n_teams)
        probs    = [c / n_sims for c in counts]
        max_prob = max(probs) if max(probs) > 0 else 1
        bmax_h   = ROW_H * 0.82
        bbot     = row_mid - bmax_h / 2
        gap      = slot_w * 0.12

        for pos_idx, prob in enumerate(probs):
            bh = (prob / max_prob) * bmax_h
            bx = X_BAR_L + pos_idx * slot_w + gap / 2
            bw = slot_w - gap
            ax.add_patch(mpatches.Rectangle(
                (bx, bbot), bw, bh,
                facecolor=_POS_COLOR[pos_idx + 1],
                edgecolor='none', zorder=1, alpha=0.88))

    # ── Section borders ───────────────────────────────────────────────────────
    vline_top = sec_bar_y + sec_bar_h + 0.04
    vline_bot = PAD_BOT - 0.10
    for vx in [SEC1_L, SEC1_R, SEC2_L, SEC2_R, X_BAR_L, X_BAR_R]:
        ax.plot([vx, vx], [vline_bot, vline_top], color='#d0d0d0', lw=0.8, zorder=5)

    plt.tight_layout(pad=0)
    return fig
