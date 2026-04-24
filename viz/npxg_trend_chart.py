"""
xg_trend_chart.py — rolling xG for/against line chart

Usage:
    from viz.xg_trend_chart import plot_xg_trend

    plot_xg_trend(
        plot_df=plot_df,          # DataFrame with columns: match_date, xg_for_roll10, xg_against_roll10, season
        title="FCK's xG collapse tells the real story",
        team="FC København",
        league="Superligaen",
        save_path="fck_xg_rolling.png",   # optional
    )
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from io import BytesIO
from scipy.stats import linregress
import requests
from PIL import Image

# Register static Roboto fonts (variable font doesn't support bold in matplotlib)
fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Regular.ttf')
fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Bold.ttf')
plt.rcParams['font.family'] = 'Roboto'


from viz.logos import TEAM_LOGOS
from viz.team_colors import TEAM_COLORS


def _fetch_logo(team_name):
    """Fetch team badge. Uses TEAM_LOGOS dict first, falls back to TheSportsDB API."""
    try:
        url = TEAM_LOGOS.get(team_name)
        if not url:
            r = requests.get(
                "https://www.thesportsdb.com/api/v1/json/3/searchteams.php",
                params={"t": team_name}, timeout=5
            )
            url = r.json()["teams"][0]["strBadge"]
        img_data = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'}).content
        return Image.open(BytesIO(img_data)).convert("RGBA")
    except Exception:
        return None


def plot_xg_trend(
    plot_df,
    title,
    team,
    league,
    save_path=None,
    figsize=(12, 5.5),
    xgf_col=None,
    xga_col=None,
    annotations=None,
    roll_n=10,
    extra_ticks=None,
):
    """
    annotations : list of (date_str, label) tuples, e.g. [('2025-10-01', "Huesca's injury")]
                  Each draws a vertical line at the nearest match to that date.
    roll_n       : rolling window size shown in the y-axis label (default 10).
    extra_ticks  : list of date strings to add as labelled x-axis ticks, e.g. ['2026-01-01'].
                   Each snaps to the nearest match date.
    """
    # Use team's primary colour for xG for, secondary for xG against if registered
    if xgf_col is None or xga_col is None:
        tc = TEAM_COLORS.get(team)
        xgf_col = xgf_col or (tc[0] if tc else '#16a34a')
        xga_col = xga_col or (tc[1] if tc else '#e63946')

    df = plot_df.sort_values("match_date").reset_index(drop=True)

    dates = df["match_date"]
    x     = np.arange(len(df))
    xgf   = df["xg_for_roll10"].values
    xga   = df["xg_against_roll10"].values

    date_range = f"{dates.iloc[0].strftime('%b %Y')} – {dates.iloc[-1].strftime('%b %Y')}"
    subtitle   = f"{team}  ·  {league}  ·  {date_range}"

    season_change_idx = [
        i for i in df[df["season"] != df["season"].shift()].index if i > 0
    ]

    def trend(x, y):
        m, b, *_ = linregress(x, y)
        return m * x + b

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.spines["left"].set_color("#cccccc")
    ax.tick_params(left=False, bottom=False)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Lines + dots ──────────────────────────────────────────────────────────
    ax.plot(x, xgf, color=xgf_col, linewidth=2, zorder=3)
    ax.scatter(x, xgf, color=xgf_col, s=28, zorder=4)
    ax.plot(x, trend(x, xgf), color=xgf_col, linewidth=1.2, linestyle="--", alpha=0.55, zorder=2)

    ax.plot(x, xga, color=xga_col, linewidth=2, zorder=3)
    ax.scatter(x, xga, color=xga_col, s=28, zorder=4)
    ax.plot(x, trend(x, xga), color=xga_col, linewidth=1.2, linestyle="--", alpha=0.55, zorder=2)

    # ── Season change vertical line ───────────────────────────────────────────
    y_min = min(np.min(xgf), np.min(xga))
    y_max = max(np.max(xgf), np.max(xga))
    y_range = y_max - y_min
    for idx in season_change_idx:
        ax.axvline(idx, color="#888888", linewidth=1.2, linestyle=":", zorder=2)
        near_right = max(xgf[idx], xga[idx]) > (y_max * 0.85)
        offset, ha = (-0.3, "right") if near_right else (0.3, "left")
        ax.text(idx + offset, y_max * 0.98, df.loc[idx, "season"],
                fontsize=8, color="#888888", va="top", ha=ha)

    # ── Custom annotations ────────────────────────────────────────────────────
    if annotations:
        import pandas as pd
        dates_series = pd.to_datetime(df["match_date"])

        # Reserve space below the data — scale buffer with the longest label
        max_chars = max(len(label) for _, label in annotations)
        y_buffer = y_range * max(0.12, 0.021 * max_chars)
        ax.set_ylim(y_min - y_buffer, y_max + y_range * 0.22)

        for date_str, label in annotations:
            target = pd.Timestamp(date_str)
            idx = (dates_series - target).abs().idxmin()
            ax.axvline(idx, color="#555555", linewidth=1.2, linestyle="--", zorder=2, alpha=0.7)
            # va='top' anchors the label top at y_min; text hangs down into the buffer
            ax.text(idx + 0.25, y_min, label,
                    fontsize=8.5, color="#555555", va="top", ha="left",
                    style="italic", rotation=90, clip_on=False)

    # ── Y-axis limits — always give headroom above the data for the legend ────
    if not annotations:
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.22)

    # ── Axes ──────────────────────────────────────────────────────────────────
    tick_positions = [0, len(df) - 1]
    tick_labels    = [dates.iloc[0].strftime("%b %Y"), dates.iloc[-1].strftime("%b %Y")]

    if extra_ticks:
        import pandas as pd
        dates_series = pd.to_datetime(df["match_date"])
        for date_str in extra_ticks:
            target = pd.Timestamp(date_str)
            idx = int((dates_series - target).abs().idxmin())
            label = target.strftime("%b %Y")
            if idx not in tick_positions:
                tick_positions.append(idx)
                tick_labels.append(label)
        # Sort by position so ticks appear left-to-right
        paired = sorted(zip(tick_positions, tick_labels))
        tick_positions, tick_labels = zip(*paired)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10, color="#444444")
    ax.set_ylabel(f"npxG ({roll_n}-game rolling avg)", fontsize=9, color="#666666", labelpad=8)
    ax.tick_params(axis="y", labelsize=9, labelcolor="#666666")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color=xgf_col, linewidth=2, label="npxG For"),
        plt.Line2D([0], [0], color=xga_col, linewidth=2, label="npxG Against"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="upper right", ncol=2,
              bbox_to_anchor=(1, 1.025
              ))

    # ── Team logo (top right of figure) ──────────────────────────────────────
    logo = _fetch_logo(team)
    if logo:
        logo.thumbnail((120, 120), Image.LANCZOS)
        logo_box = OffsetImage(logo, zoom=0.3)
        ab = AnnotationBbox(logo_box, (0.99, 0.97), xycoords='figure fraction',
                            box_alignment=(1, 1), frameon=False)
        fig.add_artist(ab)

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    # ── Title + subtitle ──────────────────────────────────────────────────────
    ax.text(0, 1.07, title,    transform=ax.transAxes, fontsize=13, fontweight="bold", color="#222222", va="bottom", ha="left")
    ax.text(0.001, 1.025, subtitle, transform=ax.transAxes, fontsize=9,  color="#888888",   va="bottom", ha="left")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.show()
