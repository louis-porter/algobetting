"""
manual_priors.py — Hand-derived Bayesian priors for the Premier League model.

These were derived from 2024-2025 season posterior estimates and are used as
informative starting points for the 2025-2026 season. Teams promoted from the
Championship (Burnley, Leeds, Sunderland) use wider sigma=0.6 to reflect greater
uncertainty. Established PL sides use sigma=0.4.

Values are the raw (pre-normalisation) mu estimates. The model normalises them
at runtime by subtracting the mean across all teams, ensuring the priors sum to
zero (identifiability constraint).

Usage
-----
    from manual_priors import MANUAL_ATT_PRIORS, MANUAL_DEF_PRIORS
    _, trace = build_and_sample_model(
        df, n_teams, team_mapping=team_mapping,
        manual_att_priors=MANUAL_ATT_PRIORS,
        manual_def_priors=MANUAL_DEF_PRIORS,
    )
"""

import numpy as np

# ── Raw priors (pre-normalisation) ───────────────────────────────────────────

_RAW_ATT = {
    'Arsenal':        (0.198, 0.4),
    'Aston Villa':    (0.121, 0.4),
    'Bournemouth':    (0.100, 0.4),
    'Brentford':      (0.079, 0.4),
    'Brighton':       (0.083, 0.4),
    'Chelsea':        (0.185, 0.4),
    'Crystal Palace': (0.057, 0.4),
    'Everton':        (-0.108, 0.4),
    'Fulham':         (-0.030, 0.4),
    'Liverpool':      (0.571, 0.4),
    'Man City':       (0.297, 0.4),
    'Man United':     (0.011, 0.4),
    'Newcastle':      (0.207, 0.4),
    'Nottm Forest':   (-0.003, 0.4),
    'Tottenham':      (0.121, 0.4),
    'West Ham':       (-0.082, 0.4),
    'Wolves':         (-0.094, 0.4),
    # Promoted sides — wider prior
    'Burnley':        (-0.401, 0.6),
    'Leeds':          (-0.103, 0.6),
    'Sunderland':     (-0.553, 0.6),
}

_RAW_DEF = {
    'Arsenal':        (-0.340, 0.4),
    'Aston Villa':    (-0.085, 0.4),
    'Bournemouth':    (-0.001, 0.4),
    'Brentford':      (-0.054, 0.4),
    'Brighton':       (0.107, 0.4),
    'Chelsea':        (-0.166, 0.4),
    'Crystal Palace': (0.052, 0.4),
    'Everton':        (-0.085, 0.4),
    'Fulham':         (-0.014, 0.4),
    'Liverpool':      (-0.222, 0.4),
    'Man City':       (-0.191, 0.4),
    'Man United':     (-0.035, 0.4),
    'Newcastle':      (-0.083, 0.4),
    'Nottm Forest':   (-0.046, 0.4),
    'Tottenham':      (0.163, 0.4),
    'West Ham':       (0.103, 0.4),
    'Wolves':         (0.137, 0.4),
    # Promoted sides — wider prior
    'Burnley':        (0.163, 0.6),
    'Leeds':          (0.163, 0.6),
    'Sunderland':     (0.231, 0.6),
}


def _normalise(raw):
    """Subtract mean of mus so priors sum to zero (identifiability)."""
    teams = list(raw.keys())
    mus   = np.array([raw[t][0] for t in teams])
    mus  -= mus.mean()
    sigs  = np.array([raw[t][1] for t in teams])
    return {t: (float(mus[i]), float(sigs[i])) for i, t in enumerate(teams)}


MANUAL_ATT_PRIORS = _normalise(_RAW_ATT)
MANUAL_DEF_PRIORS = _normalise(_RAW_DEF)
