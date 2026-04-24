"""
team_colors.py — team primary and secondary colour registry

Each entry: team_name → (primary, secondary)

Used by all viz functions to auto-resolve colours:
  - Two different teams  → primary of each team
  - Same team (e.g. season comparison) → primary + secondary of that team

Usage:
    from viz.team_colors import get_colors

    color_a, color_b = get_colors('FC København', 'FC Midtjylland')
    color_a, color_b = get_colors('FC København', 'FC København')  # primary + secondary
"""

# ── Registry ──────────────────────────────────────────────────────────────────
# Note: where a club's official colour is white, a chart-readable shade is used
# instead and noted in a comment.

TEAM_COLORS = {

    # ── Superligaen ───────────────────────────────────────────────────────────
    'FC København':     ('#0F2DB9', '#C8A84B'),   # blue, gold
    'Copenhagen':       ('#0F2DB9', '#C8A84B'),

    'FC Midtjylland':   ('#C8102E', '#1D1D1B'),   # red, black
    'Midtjylland':      ('#C8102E', '#1D1D1B'),

    'Brøndby IF':       ('#F5C800', '#003399'),   # yellow, blue
    'Brondby':          ('#F5C800', '#003399'),

    'OB':               ('#003366', '#7AB2D4'),   # dark blue, sky blue
    'Odense':           ('#003366', '#7AB2D4'),

    'Nordsjælland':     ('#E2001A', '#231F20'),   # red, black
    'Nordsjaelland':    ('#E2001A', '#231F20'),

    'AGF':              ('#8B0000', '#1B2A4A'),   # dark red, navy (away kit)
    'Aarhus':           ('#8B0000', '#B0B0B0'),

    'Randers FC':       ('#002D72', '#6CABDD'),   # dark blue, sky blue
    'Randers':          ('#002D72', '#6CABDD'),

    'Vejle Boldklub':   ('#CC0000', '#1D1D1B'),   # red, black
    'Vejle':            ('#CC0000', '#1D1D1B'),

    'Sønderjyske':      ('#003087', '#CC0000'),   # blue, red
    'Sonderjyske':      ('#003087', '#CC0000'),

    'Silkeborg IF':     ('#003399', '#CC0000'),   # blue, red
    'Silkeborg':        ('#003399', '#CC0000'),

    'Viborg FF':        ('#005C2E', '#7AB648'),   # dark green, light green
    'Viborg':           ('#005C2E', '#7AB648'),

    'Fredericia':       ('#CC0000', '#1D1D1B'),   # red, black

    # ── Premier League ────────────────────────────────────────────────────────
    'Arsenal':          ('#EF0107', '#063672'),   # red, navy
    'Aston Villa':      ('#670E36', '#95BFE5'),   # claret, sky blue
    'Bournemouth':      ('#DA291C', '#231F20'),   # red, black
    'Brentford':        ('#E30613', '#FFE500'),   # red, yellow
    'Brighton':         ('#0057B8', '#FFCD00'),   # blue, yellow
    'Chelsea':          ('#034694', '#DBA111'),   # blue, gold
    'Crystal Palace':   ('#1B458F', '#C4122E'),   # blue, red
    'Everton':          ('#003399', '#A0A0A0'),   # blue, white (white used sparingly)
    'Fulham':           ('#231F20', '#CC0000'),   # black, red
    'Liverpool':        ('#C8102E', '#00B2A9'),   # red, teal
    'Man City':         ('#6CABDD', '#1C2C5B'),   # sky blue, dark blue
    'Man United':       ('#DA291C', '#FBE122'),   # red, yellow
    'Newcastle':        ('#241F20', '#A0A0A0'),   # black, grey (white → grey for charts)
    'Nottm Forest':     ('#DD0000', '#B0B0B0'),   # red, grey (white → grey)
    'Sunderland':       ('#EB172B', '#231F20'),   # red, black
    'Tottenham':        ('#132257', '#A0A0A0'),   # navy, grey (white → grey)
    'West Ham':         ('#7A263A', '#1BB1E7'),   # claret, sky blue
    'Wolves':           ('#FDB913', '#231F20'),   # gold, black

    # ── Championship ─────────────────────────────────────────────────────────
    'Burnley':          ('#6C1D45', '#91BCD9'),   # claret, sky blue
    'Leeds United':     ('#FFD100', '#1D428A'),   # blue, yellow (white → blue)
    'Leeds':            ('#FFD100', '#1D428A'),
}

_DEFAULT_A = '#457b9d'
_DEFAULT_B = '#e63946'


def _hex_to_hsl(hex_color):
    import colorsys
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    return hue * 360, sat * 100, light * 100   # (hue 0-360, sat 0-100, light 0-100)


def _hue_distance(h1, h2):
    diff = abs(h1 - h2)
    return min(diff, 360 - diff)


def colors_clash(color_a, color_b, hue_threshold=40):
    """
    Return True if two hex colours are too similar to distinguish on a chart.

    Rules:
    - Both chromatic → clash if hue distance < hue_threshold (same colour family)
    - Both near-grey  → clash if lightness is similar (two dark or two light greys)
    - One chromatic, one near-grey → never a clash (always visually distinct)
    """
    h1, s1, l1 = _hex_to_hsl(color_a)
    h2, s2, l2 = _hex_to_hsl(color_b)

    a_grey = s1 < 15
    b_grey = s2 < 15

    if a_grey and b_grey:
        return abs(l1 - l2) < 20   # both achromatic: clash if similar lightness
    if a_grey or b_grey:
        return False                # one chromatic, one not: always distinct
    return _hue_distance(h1, h2) < hue_threshold


def get_colors(team_a, team_b=None):
    """
    Resolve chart colours for a team comparison.

    Parameters
    ----------
    team_a : str
        First team name (must match a key in TEAM_COLORS).
    team_b : str or None
        Second team. If None or same as team_a, uses team_a's secondary colour.

    Returns
    -------
    (color_a, color_b) : tuple of hex strings
    """
    ca = TEAM_COLORS.get(team_a)
    cb = TEAM_COLORS.get(team_b) if team_b else None

    same_team = (team_b is None) or (team_a == team_b)

    if same_team:
        if ca:
            return ca[0], ca[1]
        return _DEFAULT_A, _DEFAULT_B
    else:
        color_a = ca[0] if ca else _DEFAULT_A
        color_b = cb[0] if cb else _DEFAULT_B
        # If primaries clash, fall back to team_b's secondary
        if cb and colors_clash(color_a, color_b):
            color_b = cb[1]
        return color_a, color_b
