import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from src.trace_save_load import save_season_trace

def build_and_sample_model(train_df, n_teams, current_season=None, league=None, 
                          trace=5000, tune=2500, team_mapping=None, model_version="v1",
                          manual_att_priors=None, manual_def_priors=None,
                          red_card_priors=None):
    """Build and sample the football model with manual team priors and red card effects
    
    Parameters:
    -----------
    manual_att_priors : dict or list, optional
        Manual attack strength priors for teams. Can be:
        - dict: {team_name: (mu, sigma), ...} where team_name matches keys in team_mapping
        - list: [(mu, sigma), ...] in same order as team indices
    manual_def_priors : dict or list, optional
        Manual defense strength priors for teams. Same format as manual_att_priors
    red_card_priors : dict, optional
        Priors for red card effects derived from historical analysis. Format:
        {
            'att_effect_mu': float,     # e.g., -0.51 (log scale, ~-40% goals)
            'att_effect_sigma': float,  # e.g., 0.15
            'def_effect_mu': float,     # e.g., 0.26 (log scale, ~+30% conceded)
            'def_effect_sigma': float   # e.g., 0.15
        }
        If None, uses weakly informative priors
    
    Expected columns in train_df for red card modeling:
    -----------
    home_red_proportion : float
        Proportion of match home team played with red card (0.0 to 1.0)
    away_red_proportion : float
        Proportion of match away team played with red card (0.0 to 1.0)
    """
    
    home_idx = train_df['home_idx'].values
    away_idx = train_df['away_idx'].values

    home_goals_obs = train_df["home_goals"]
    away_goals_obs = train_df["away_goals"]
    
    # Red card proportions (will be 0 if no red cards in that match)
    home_red_prop = train_df.get('home_red_proportion', pd.Series(np.zeros(len(train_df)))).values
    away_red_prop = train_df.get('away_red_proportion', pd.Series(np.zeros(len(train_df)))).values
    
    # Set default red card priors if not provided
    if red_card_priors is None:
        red_card_priors = {
            'att_effect_mu': -0.60,      # ~-40% scoring reduction
            'att_effect_sigma': 0.2,     # Allow uncertainty
            'def_effect_mu': 0.55,       # ~+28% more goals conceded
            'def_effect_sigma': 0.2      # Allow uncertainty
        }
   
    def process_manual_priors(priors, n_teams, team_mapping):
        """Convert manual priors to arrays of mu and sigma values"""
        if priors is None:
            return None, None
            
        mu_array = np.zeros(n_teams)
        sigma_array = np.ones(n_teams)  # Default sigma = 1
        
        if isinstance(priors, dict):
            if team_mapping is None:
                raise ValueError("team_mapping must be provided when using dict priors")
            
            for team_name, (mu, sigma) in priors.items():
                if team_name in team_mapping:
                    idx = team_mapping[team_name]
                    mu_array[idx] = mu
                    sigma_array[idx] = sigma
                else:
                    print(f"Warning: Team '{team_name}' not found in team_mapping")
        
        return mu_array, sigma_array
        
        
    with pm.Model() as model:
        # Process manual priors
        att_mu, att_sigma = process_manual_priors(manual_att_priors, n_teams, team_mapping)
        def_mu, def_sigma = process_manual_priors(manual_def_priors, n_teams, team_mapping)

        # Hierarchical priors - let the data learn the spread of team abilities
        sigma_att = pm.HalfNormal("sigma_att", sigma=1.5)
        sigma_def = pm.HalfNormal("sigma_def", sigma=1.5)

        # Set up attack strength priors
        if att_mu is not None:
            att_str_raw = pm.Normal("att_str_raw", mu=att_mu, sigma=att_sigma, shape=n_teams)
        else:
            att_str_raw = pm.Normal("att_str_raw", mu=0, sigma=sigma_att, shape=n_teams)
            
        # Set up defense strength priors  
        if def_mu is not None:
            def_str_raw = pm.Normal("def_str_raw", mu=def_mu, sigma=def_sigma, shape=n_teams)
        else:
            def_str_raw = pm.Normal("def_str_raw", mu=0, sigma=sigma_def, shape=n_teams)

        # Center defense only (standard practice)
        att_str = pm.Deterministic("att_str", att_str_raw - pm.math.mean(att_str_raw))
        def_str = pm.Deterministic("def_str", def_str_raw - pm.math.mean(def_str_raw))
        
        # RED CARD EFFECTS - learned from historical data but updatable with current season
        # Negative = team scores less when down a man
        red_att_effect = pm.Normal(
            "red_att_effect", 
            mu=red_card_priors['att_effect_mu'],
            sigma=red_card_priors['att_effect_sigma']
        )
        
        # Positive = team concedes more when down a man  
        red_def_effect = pm.Normal(
            "red_def_effect",
            mu=red_card_priors['def_effect_mu'], 
            sigma=red_card_priors['def_effect_sigma']
        )
        
        # Other model components
        home_adv = pm.Normal("home_adv", mu=np.log(1.21), sigma=0.1)
        baseline = pm.Normal("baseline", mu=np.log(1.31), sigma=0.1)

        # ADJUSTED GOAL RATES WITH RED CARD EFFECTS
        # Home team scoring: reduced if home has red, increased if away has red (weaker defense)
        home_goals_mu = pm.math.exp(
            baseline + 
            att_str[home_idx] + 
            def_str[away_idx] + 
            home_adv +
            home_red_prop * red_att_effect +      # Home attacking impaired
            away_red_prop * (-red_def_effect)     # Away defending impaired (helps home score)
        )
        
        # Away team scoring: reduced if away has red, increased if home has red (weaker defense)
        away_goals_mu = pm.math.exp(
            baseline + 
            att_str[away_idx] + 
            def_str[home_idx] +
            away_red_prop * red_att_effect +      # Away attacking impaired
            home_red_prop * (-red_def_effect)     # Home defending impaired (helps away score)
        )

        try:
            weights = pm.ConstantData("weights", train_df["weight"].values)
        except AttributeError:
            # For older PyMC versions
            weights = pm.Data("weights", train_df["weight"].values)
            
        home_logp = pm.logp(pm.Poisson.dist(mu=home_goals_mu), home_goals_obs)
        away_logp = pm.logp(pm.Poisson.dist(mu=away_goals_mu), away_goals_obs)
        pm.Potential("weighted_home_goals", pm.math.sum(weights * home_logp))
        pm.Potential("weighted_away_goals", pm.math.sum(weights * away_logp))

        # Sample
        trace = pm.sample(trace=trace, tune=tune, cores=4, nuts_sampler='blackjax', 
                         return_inferencedata=True, progressbar=False)
        
    return model, trace