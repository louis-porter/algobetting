import numpy as np
from scipy.stats import poisson


def _goal_grid(home_lam: float, away_lam: float, max_goals: int = 14) -> np.ndarray:
    h = poisson.pmf(np.arange(max_goals + 1), home_lam)
    a = poisson.pmf(np.arange(max_goals + 1), away_lam)
    return np.outer(h, a)


def _ah_probs(grid: np.ndarray, line: float):
    """Return (p_home_cover, p_away_cover) for a single AH line."""
    frac = round(line % 1, 10)
    if frac in (0.25, 0.75):
        lo = np.floor(line * 2) / 2
        ph_lo, pa_lo = _ah_probs(grid, lo)
        ph_hi, pa_hi = _ah_probs(grid, lo + 0.5)
        return (ph_lo + ph_hi) / 2, (pa_lo + pa_hi) / 2

    n = grid.shape[0]
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    adj = (i - j) + line   # positive = home covers

    p_home = float(grid[adj > 0].sum())
    p_push = float(grid[adj == 0].sum())
    p_away = float(grid[adj < 0].sum())

    if p_push > 1e-9:
        denom = 1 - p_push
        return p_home / denom, p_away / denom
    return p_home, p_away


def price_asian_handicap(home_lam: float, away_lam: float, max_goals: int = 14):
    """
    Price all AH lines from -4 to +4 in 0.25 steps (home perspective).
    Negative line = home favoured (gives goals); positive = away favoured.

    Returns a list of 3 dicts for the most balanced line and its immediate
    neighbours (one step tighter, one step looser from home's perspective):

        [{'line': float, 'home_odds': float, 'away_odds': float}, ...]

    home_odds / away_odds are fair decimal odds (no margin).
    """
    grid  = _goal_grid(home_lam, away_lam, max_goals)
    lines = [round(x * 0.25, 2) for x in range(-16, 17)]  # -4.0 … +4.0

    scored = []
    for line in lines:
        ph, pa = _ah_probs(grid, line)
        scored.append({
            'line':      line,
            'home_odds': round(1 / ph, 2),
            'away_odds': round(1 / pa, 2),
            'balance':   abs(ph - 0.5),
        })

    best = min(range(len(scored)), key=lambda i: scored[i]['balance'])
    best = max(1, min(best, len(scored) - 2))   # keep neighbours in bounds

    return [
        {'line': r['line'], 'home_odds': r['home_odds'], 'away_odds': r['away_odds']}
        for r in scored[best - 1 : best + 2]
    ]
