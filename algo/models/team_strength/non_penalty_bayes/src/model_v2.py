import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd

def build_and_sample_dynamic_model(
    expanded_df: pd.DataFrame,
    n_teams: int,
    n_periods: int,
    draws: int = 2000,
    tune: int = 1000,
    cores: int = 4,
    promoted_init_means: np.ndarray = None,
    promoted_init_sigmas: np.ndarray = None,
    random_seed: int = 42,
    sampler_kwargs: dict = None,
):
    """
    Dynamic team-strength model (random walk) with weighted scoreline likelihood.
    Red cards are assumed to already be included in the dataset weights.
    """

    # Match-level arrays (one row per candidate scoreline)
    home_idx = expanded_df['home_idx'].astype('int64').values
    away_idx = expanded_df['away_idx'].astype('int64').values
    period_idx = expanded_df['period'].astype('int64').values
    home_goals = expanded_df['home_goals'].astype('int64').values
    away_goals = expanded_df['away_goals'].astype('int64').values
    weights = expanded_df['weight'].astype('float64').values

    n_obs = len(expanded_df)
    if sampler_kwargs is None:
        sampler_kwargs = {}

    with pm.Model() as model:
        # ----------------------
        # Hyperpriors
        # ----------------------
        sigma_att_rw = pm.HalfNormal("sigma_att_rw", sigma=0.8)
        sigma_def_rw = pm.HalfNormal("sigma_def_rw", sigma=0.8)

        baseline = pm.Normal("baseline", mu=0.0, sigma=1.0)
        home_adv = pm.Normal("home_adv", mu=0.15, sigma=0.2)

        # ----------------------
        # Initial strengths
        # ----------------------
        if promoted_init_means is None:
            promoted_init_means = np.zeros(n_teams)
        if promoted_init_sigmas is None:
            promoted_init_sigmas = np.ones(n_teams)

        att_init = pm.Normal("att_init", mu=promoted_init_means,
                             sigma=promoted_init_sigmas, shape=n_teams)
        def_init = pm.Normal("def_init", mu=promoted_init_means,
                             sigma=promoted_init_sigmas, shape=n_teams)

        # ----------------------
        # Random walk evolution
        # ----------------------
        if n_periods < 2:
            att = att_init[:, None]
            defe = def_init[:, None]
        else:
            att_incr = pm.Normal("att_incr", mu=0.0, sigma=sigma_att_rw,
                                 shape=(n_teams, n_periods - 1))
            def_incr = pm.Normal("def_incr", mu=0.0, sigma=sigma_def_rw,
                                 shape=(n_teams, n_periods - 1))

            # Changed from at.concatenate to pt.concatenate
            att = pt.concatenate(
                [att_init[:, None], att_init[:, None] + pt.cumsum(att_incr, axis=1)],
                axis=1
            )
            defe = pt.concatenate(
                [def_init[:, None], def_init[:, None] + pt.cumsum(def_incr, axis=1)],
                axis=1
            )

        # ----------------------
        # Identifiability: center per period
        # ----------------------
        # Changed from at.mean to pt.mean
        att_centered = att - pt.mean(att, axis=0, keepdims=True)
        def_centered = defe - pt.mean(defe, axis=0, keepdims=True)

        # ----------------------
        # Match-specific strengths
        # ----------------------
        home_att = att_centered[home_idx, period_idx]
        away_att = att_centered[away_idx, period_idx]
        home_def = def_centered[home_idx, period_idx]
        away_def = def_centered[away_idx, period_idx]

        log_mu_home = baseline + home_att + away_def + home_adv
        log_mu_away = baseline + away_att + home_def

        # Changed from at.exp to pt.exp
        mu_home = pt.exp(log_mu_home)
        mu_away = pt.exp(log_mu_away)

        # ----------------------
        # Weighted likelihood
        # ----------------------
        pm.ConstantData("weights", weights)

        home_logp = pm.logp(pm.Poisson.dist(mu=mu_home), home_goals)
        away_logp = pm.logp(pm.Poisson.dist(mu=mu_away), away_goals)

        # Changed from at.sum to pt.sum
        pm.Potential("weighted_likelihood", pt.sum(weights * (home_logp + away_logp)))

        # ----------------------
        # Sampling
        # ----------------------
        idata = pm.sample(draws=draws, tune=tune, cores=cores,
                          random_seed=random_seed, return_inferencedata=True,
                          progressbar=False, nuts_sampler="blackjax")

    return model, idata