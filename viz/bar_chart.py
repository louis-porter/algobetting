"""
bar_chart.py — Datawrapper-style horizontal bar chart with team logos

Usage:
    from viz.bar_chart import plot_bar_chart

    data = {
        "Copenhagen":    21_700_000,
        "Midtjylland":   14_600_000,
        ...
    }

    plot_bar_chart(
        data=data,
        title="Copenhagen Spend More Than Double Their Nearest Rival",
        subtitle="2026 squad payroll, Superligaen",
        value_format="€{:,.0f}",
        save_path="superligaen_payroll.png",   # optional
    )

    # CSV export for Datawrapper table:
    from viz.bar_chart import datawrapper_csv
    datawrapper_csv(data, value_label="2026 Payroll")
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import requests
from io import BytesIO
from PIL import Image

from viz.logos import TEAM_LOGOS
from viz.team_colors import get_colors, TEAM_COLORS

# Register Roboto
fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Regular.ttf')
fm.fontManager.addfont('/Users/admin/Library/Fonts/Roboto-Bold.ttf')
plt.rcParams['font.family'] = 'Roboto'


def _fetch_logo(team_name):
    try:
        url = TEAM_LOGOS.get(team_name)
        if not url:
            r = requests.get(
                "https://www.thesportsdb.com/api/v1/json/3/searchteams.php",
                params={"t": team_name}, timeout=5
            )
            url = r.json()["teams"][0]["strBadge"]
        img_data = requests.get(url, timeout=5).content
        return np.array(Image.open(BytesIO(img_data)).convert("RGBA"))
    except Exception:
        return None


def plot_bar_chart(
    data,
    title,
    subtitle,
    value_format="{:,.0f}",
    bar_col="#457b9d",
    highlight_team=None,
    highlight_col="#e63946",
    save_path=None,
    figsize=None,
):
    teams  = list(data.keys())
    values = list(data.values())
    n      = len(teams)

    # Auto-resolve highlight colour from team registry if not overridden
    if highlight_team and highlight_col == '#e63946' and highlight_team in TEAM_COLORS:
        highlight_col = TEAM_COLORS[highlight_team][0]

    figsize = figsize or (10, max(4, n * 0.6))
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    max_val = max(values)
    y = np.arange(n)

    # ── Bars ──────────────────────────────────────────────────────────────────
    colors = [
        highlight_col if highlight_team and t == highlight_team else bar_col
        for t in teams
    ]
    ax.barh(y, values, color=colors, height=0.55, zorder=3)

    # ── Value labels ──────────────────────────────────────────────────────────
    for i, val in enumerate(values):
        ax.text(
            val + max_val * 0.01, i,
            value_format.format(val),
            va="center", ha="left", fontsize=9, color="#444444"
        )

    # ── Logos via inset_axes (renders at full figure DPI) ─────────────────────
    ax.set_yticks(y)
    ax.set_yticklabels([""] * n)
    ax.set_xlim(-max_val * 0.42, max_val * 1.18)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()

    fig.canvas.draw()  # needed to compute transforms before placing insets

    # Axes pixel dimensions — needed to make logos square in display space
    ax_bbox  = ax.get_window_extent()
    ax_w_px  = ax_bbox.width
    ax_h_px  = ax_bbox.height

    for i, team in enumerate(teams):
        logo = _fetch_logo(team)

        # Team name label
        ax.text(-max_val * 0.30, i, team, va="center", ha="left",
                fontsize=10, color="#222222")

        if logo is None:
            continue

        # Convert data coords of logo centre to axes fraction
        logo_data_x = -max_val * 0.37
        disp = ax.transData.transform((logo_data_x, i))
        ax_frac_x, ax_frac_y = ax.transAxes.inverted().transform(disp)

        # Make logo square in pixel space
        logo_h_frac = 0.65 / n          # scale with number of rows
        logo_h_px   = logo_h_frac * ax_h_px
        logo_w_frac = logo_h_px / ax_w_px

        axin = ax.inset_axes(
            [ax_frac_x - logo_w_frac / 2, ax_frac_y - logo_h_frac / 2,
             logo_w_frac, logo_h_frac]
        )
        axin.imshow(logo, interpolation="lanczos")
        axin.axis("off")

    # ── Style ─────────────────────────────────────────────────────────────────
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks([])

    # ── Title + subtitle ──────────────────────────────────────────────────────
    plt.tight_layout()
    fig.subplots_adjust(top=0.91, left=0.01)
    ax.text(0, 1.07, title,    transform=ax.transAxes, fontsize=13,
            fontweight="bold", color="#222222", va="bottom")
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=9,
            color="#888888", va="bottom")

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


def datawrapper_csv(data, value_label="Value"):
    """Print a CSV with logo markdown for pasting into Datawrapper."""
    import io, csv
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Logo", "Team", value_label])
    for team, val in data.items():
        logo_url = TEAM_LOGOS.get(team, "")
        logo_md  = f"![]({logo_url})" if logo_url else ""
        w.writerow([logo_md, team, val])
    print("--- COPY EVERYTHING BELOW THIS LINE ---")
    print(out.getvalue())
