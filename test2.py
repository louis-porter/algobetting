import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def prepare_data(df):
    """Prepare data for the hierarchical model"""
    # Get unique teams and create mappings
    teams = sorted(df['team'].unique())
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    
    # Create indices for home and away teams
    home_idx = df['team'].map(team_to_idx).values
    away_idx = df['opp_team'].map(team_to_idx).values
    
    # Convert to numpy arrays
    goals_home = df['goals_for'].values
    goals_away = df['goals_against'].values
    xg_home = df['xg_for'].values
    xg_away = df['xg_against'].values
    shots_home = df['shots_for'].values
    shots_away = df['shots_against'].values
    
    return {
        'teams': teams,
        'n_teams': n_teams,
        'n_matches': len(df),
        'home_idx': home_idx,
        'away_idx': away_idx,
        'goals_home': goals_home,
        'goals_away': goals_away,
        'xg_home': xg_home,
        'xg_away': xg_away,
        'shots_home': shots_home,
        'shots_away': shots_away
    }

def build_model(data):
    """Build the hierarchical Bayesian model"""
    
    with pm.Model() as model:
        # Level 3: Underlying team abilities for performance metrics
        # xG generation ability (log scale)
        mu_xg_att = pm.Normal('mu_xg_att', mu=0.3, sigma=0.3, shape=data['n_teams'])  # ~exp(0.3) = 1.35 xG
        mu_xg_def = pm.Normal('mu_xg_def', mu=0.3, sigma=0.3, shape=data['n_teams'])  # xG allowed
        
        # Shots generation ability (log scale) 
        mu_shots_att = pm.Normal('mu_shots_att', mu=2.5, sigma=0.3, shape=data['n_teams'])  # ~exp(2.5) = 12 shots
        mu_shots_def = pm.Normal('mu_shots_def', mu=2.5, sigma=0.3, shape=data['n_teams'])  # shots allowed
        
        # Level 2: Attack and Defence parameters from performance metrics
        # Coefficients for how xG and shots translate to attack/defence strength
        beta_xg = pm.Normal('beta_xg', mu=1.0, sigma=0.3)      # xG coefficient for attack
        beta_shots = pm.Normal('beta_shots', mu=0.2, sigma=0.1) # shots coefficient for attack
        
        gamma_xg = pm.Normal('gamma_xg', mu=1.0, sigma=0.3)     # xG coefficient for defence  
        gamma_shots = pm.Normal('gamma_shots', mu=0.2, sigma=0.1) # shots coefficient for defence
        
        # Intercepts
        att_intercept = pm.Normal('att_intercept', mu=0, sigma=0.2)
        def_intercept = pm.Normal('def_intercept', mu=0, sigma=0.2)
        
        # Attack and defence parameters
        attack = pm.Deterministic('attack', 
                                att_intercept + beta_xg * mu_xg_att + beta_shots * mu_shots_att)
        defence = pm.Deterministic('defence', 
                                 def_intercept + gamma_xg * mu_xg_def + gamma_shots * mu_shots_def)
        
        # Home advantage
        home_adv = pm.Normal('home_adv', mu=0.1, sigma=0.1)
        
        # Level 1: Goal generation from attack/defence
        # Expected goals for home and away teams
        lambda_home = pm.math.exp(attack[data['home_idx']] - defence[data['away_idx']] + home_adv)
        lambda_away = pm.math.exp(attack[data['away_idx']] - defence[data['home_idx']])
        
        # Observed goals (Poisson likelihood)
        goals_home_obs = pm.Poisson('goals_home_obs', mu=lambda_home, observed=data['goals_home'])
        goals_away_obs = pm.Poisson('goals_away_obs', mu=lambda_away, observed=data['goals_away'])
        
        # Level 3: Performance metrics observations
        # These constrain the underlying abilities
        xg_home_obs = pm.Normal('xg_home_obs', 
                               mu=pm.math.exp(mu_xg_att[data['home_idx']]), 
                               sigma=0.3, observed=data['xg_home'])
        xg_away_obs = pm.Normal('xg_away_obs', 
                               mu=pm.math.exp(mu_xg_att[data['away_idx']]), 
                               sigma=0.3, observed=data['xg_away'])
        
        shots_home_obs = pm.Normal('shots_home_obs', 
                                  mu=pm.math.exp(mu_shots_att[data['home_idx']]), 
                                  sigma=2.0, observed=data['shots_home'])
        shots_away_obs = pm.Normal('shots_away_obs', 
                                  mu=pm.math.exp(mu_shots_att[data['away_idx']]), 
                                  sigma=2.0, observed=data['shots_away'])
        
        # Defence metrics (xG and shots allowed)
        xg_def_home_obs = pm.Normal('xg_def_home_obs', 
                                   mu=pm.math.exp(mu_xg_def[data['home_idx']]), 
                                   sigma=0.3, observed=data['xg_away'])  # Home defense vs away attack
        xg_def_away_obs = pm.Normal('xg_def_away_obs', 
                                   mu=pm.math.exp(mu_xg_def[data['away_idx']]), 
                                   sigma=0.3, observed=data['xg_home'])  # Away defense vs home attack
        
        shots_def_home_obs = pm.Normal('shots_def_home_obs', 
                                      mu=pm.math.exp(mu_shots_def[data['home_idx']]), 
                                      sigma=2.0, observed=data['shots_away'])
        shots_def_away_obs = pm.Normal('shots_def_away_obs', 
                                      mu=pm.math.exp(mu_shots_def[data['away_idx']]), 
                                      sigma=2.0, observed=data['shots_home'])
    
    return model

def fit_model(model, draws=1000, tune=1000):
    """Fit the model using MCMC sampling"""
    with model:
        # Use single core to avoid Windows multiprocessing issues
        trace = pm.sample(draws=draws, tune=tune, cores=1, random_seed=42)
    return trace

def analyze_results(trace, data):
    """Analyze and visualize results"""
    # Get team rankings by attack and defence
    attack_mean = trace.posterior['attack'].mean(dim=['chain', 'draw']).values
    defence_mean = trace.posterior['defence'].mean(dim=['chain', 'draw']).values
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'team': data['teams'],
        'attack': attack_mean,
        'defence': defence_mean,
        'net_strength': attack_mean - defence_mean
    }).sort_values('net_strength', ascending=False)
    
    print("Team Rankings by Net Strength:")
    print(results_df)
    
    # Print key coefficients
    print(f"\nModel Coefficients:")
    print(f"xG coefficient (attack): {trace.posterior['beta_xg'].mean():.3f}")
    print(f"Shots coefficient (attack): {trace.posterior['beta_shots'].mean():.3f}")
    print(f"Home advantage: {trace.posterior['home_adv'].mean():.3f}")
    
    return results_df

# Example usage:
if __name__ == '__main__':
    # Assuming you have your df ready:
    import sqlite3
    from multiprocessing import freeze_support
    freeze_support()  # For Windows compatibility

    conn = sqlite3.connect(r'C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db')

    # Get all matches for both home and away
    df = pd.read_sql_query("""
                SELECT 
                    team,
                    opp_team,
                    summary_goals as goals_for,
                    opp_summary_goals as goals_against,
                    summary_xg as xg_for,
                    opp_summary_xg as xg_against,
                    summary_shots as shots_for,
                    opp_summary_shots as shots_against,
                    match_date as date
                FROM fbref_match_all_columns
                WHERE division = 'Premier League'
                    AND season = '2024-2025'
                    AND summary_xg IS NOT NULL
                    AND opp_summary_xg IS NOT NULL
                    AND is_home = 1

            """, conn)
    conn.close()

    data = prepare_data(df)
    model = build_model(data)
    trace = fit_model(model, draws=500, tune=500)  # Start with smaller numbers for testing
    results = analyze_results(trace, data)

# Quick model diagnostics
def plot_diagnostics(trace):
    """Plot basic model diagnostics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Trace plots for key parameters
    az.plot_trace(trace, var_names=['beta_xg', 'beta_shots'], axes=axes)
    plt.tight_layout()
    plt.show()
    
    # R-hat values
    print("R-hat diagnostics (should be < 1.1):")
    print(az.rhat(trace).max())