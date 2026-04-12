"""
pl_html_export.py — Substack HTML power rankings export.

Usage
-----
    from pl_html_export import render_html_export

    render_html_export(
        form_merged   = form_merged,    # DataFrame index=team, cols: net_rating/form_rank/prev_form_rank
        standings     = standings,      # DataFrame cols: team/pts/gd/gf/table_pos
        ratings_ranked = ratings_ranked, # DataFrame cols: team/goals_for/att_rank/goals_against/def_rank
        team_logos    = team_logos,     # dict of team → badge URL
        df_actual     = df_actual,      # actual results df for form/last result
        form_end      = '2026-04-06',   # date string for header
        output_path   = 'substack_power_rankings.html',
    )
"""

import re
import numpy as np
import pandas as pd


def render_html_export(form_merged, standings, ratings_ranked, team_logos,
                       df_actual, form_end, output_path='substack_power_rankings.html',
                       get_last_result=None, get_form_string=None, rank_arrow=None,
                       next_fixture_map=None):
    """
    Generate a Substack-ready HTML power rankings file.

    Parameters
    ----------
    form_merged     : DataFrame  index=team, cols net_rating / form_rank / prev_form_rank
    standings       : DataFrame  cols team / pts / gd / gf / table_pos
    ratings_ranked  : DataFrame  cols team / goals_for / att_rank / goals_against / def_rank
    team_logos      : dict       team → badge URL (unused in HTML text, included for future use)
    df_actual       : DataFrame  actual match results (passed to get_last_result / get_form_string)
    form_end        : str        date string shown in header (e.g. '2026-04-06')
    output_path     : str        where to save the HTML file
    get_last_result : callable   get_last_result(df_actual, team) → str  (imported from simulation)
    get_form_string : callable   get_form_string(df_actual, team, n=5) → str
    rank_arrow      : callable   rank_arrow(current, prev) → str (▲/▼/▬/●)
    next_fixture_map: dict       team → {opponent, team_xg, opponent_xg, home_away}
                                 Built from upcoming fixture predictions. If None, shows 'TBC'.

    Returns
    -------
    str  path to the saved file
    """
    # If helpers not provided, define simple fallbacks
    if get_last_result is None:
        def get_last_result(df, team):
            return '?'

    if get_form_string is None:
        def get_form_string(df, team, n=5):
            return '?'

    if rank_arrow is None:
        def rank_arrow(curr, prev):
            if prev is None:
                return '●'
            if curr < prev:
                return '▲'
            if curr > prev:
                return '▼'
            return '▬'

    ordinal = lambda n: (
        f"{n}{'th' if 11 <= n <= 13 else {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th')}"
    )

    def to_html_line(line):
        return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)

    lines = []
    lines.append(f"*Power Rankings — GW ending {form_end}*\n")

    for rank, (team, row) in enumerate(form_merged.iterrows(), start=1):
        prev_rank = (int(row['prev_form_rank'])
                     if not pd.isna(row.get('prev_form_rank', np.nan)) else None)
        arrow = rank_arrow(rank, prev_rank)

        st_row    = standings[standings['team'] == team]
        pts       = int(st_row['pts'].values[0])       if not st_row.empty else '?'
        table_pos = int(st_row['table_pos'].values[0]) if not st_row.empty else '?'
        table_str = ordinal(table_pos) if isinstance(table_pos, int) else table_pos

        rat_row      = ratings_ranked[ratings_ranked['team'] == team]
        att_val      = f"{rat_row['goals_for'].values[0]:.2f}"     if not rat_row.empty else '?'
        att_rank_str = ordinal(int(rat_row['att_rank'].values[0])) if not rat_row.empty else '?'
        def_val      = f"{rat_row['goals_against'].values[0]:.2f}" if not rat_row.empty else '?'
        def_rank_str = ordinal(int(rat_row['def_rank'].values[0])) if not rat_row.empty else '?'

        last_result = get_last_result(df_actual, team)
        form_str    = get_form_string(df_actual, team)
        prev_str    = str(prev_rank) if prev_rank else 'N/A'

        # Build next fixture string: Home (xg) - (xg) Away, team bolded
        fx = (next_fixture_map or {}).get(team)
        if fx:
            h     = fx['home_team']
            a     = fx['away_team']
            h_xg  = f"{fx['home_xg']:.2f}"
            a_xg  = f"{fx['away_xg']:.2f}"
            home_part = f"**{h} ({h_xg})**" if fx['home_away'] == 'H' else f"{h} ({h_xg})"
            away_part = f"**({a_xg}) {a}**" if fx['home_away'] == 'A' else f"({a_xg}) {a}"
            next_gw_str = f"{home_part} - {away_part}"
        else:
            next_gw_str = 'TBC'

        block = (
            f"**{rank}. {team} ({pts}pts, {table_str}) {arrow}**\n"
            f"**Last Ranking:** &nbsp;{prev_str}\n"
            f"**GW Result:** &nbsp;{last_result}\n"
            f"**BEST Attack Rating:** &nbsp;{att_val} ({att_rank_str})\n"
            f"**BEST Defence Rating:** &nbsp;{def_val} ({def_rank_str})\n"
            f"**PL Form:** &nbsp;{form_str}\n"
            f"**Next GW:** &nbsp;{next_gw_str}\n"
        )
        lines.append(block)

    html_blocks = []
    for block in lines:
        parts  = block.split('\n')
        header = re.sub(r'\*\*(.+?)\*\*', r'\1', parts[0])
        rest   = [to_html_line(l) for l in parts[1:]]
        html_blocks.append(f'<h3>{header}</h3>' + '<br>'.join(rest))

    html_body = '<br><br>'.join(html_blocks)

    html_output = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>
  body {{ font-family: Georgia, serif; font-size: 16px; line-height: 1.6;
         max-width: 700px; margin: 40px auto; padding: 0 20px; }}
</style></head>
<body>
{html_body}
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html_output)

    print(f"Saved to {output_path}")
    print("Open in a browser → Select All (Cmd+A) → Copy (Cmd+C) → Paste into Substack.")
    return output_path
