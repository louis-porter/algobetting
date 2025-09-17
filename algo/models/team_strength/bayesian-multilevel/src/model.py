import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from src.trace_save_load import save_season_trace

def build_and_sample_model(train_df, n_teams, 
                          trace=5000, tune=2500):
    """Build and sample the football model with league-specific parameters"""
    
    home_idx = train_df['home_idx'].values
    away_idx = train_df['away_idx'].values
    home_goals_obs = train_df['home_goals'].values
    away_goals_obs = train_df['away_goals'].values
    
    # League indicators (1=Premier League, 0=Championship)
    is_prem = (train_df['league_id'] == 'Premier_League').astype(float).values
    
    with pm.Model() as model:
        # Team strength priors
        att_str_raw = pm.Normal("att_str_raw", mu=0, sigma=1, shape=n_teams)
        def_str_raw = pm.Normal("def_str_raw", mu=0, sigma=1, shape=n_teams)
        
        # Apply sum-to-zero constraints
        att_str = att_str_raw#pm.Deterministic("att_str", att_str_raw - pm.math.mean(att_str_raw))
        def_str = def_str_raw#pm.Deterministic("def_str", def_str_raw - pm.math.mean(def_str_raw))
        
        # League-specific parameters
        baseline_prem = pm.Normal("baseline_prem", mu=0.37, sigma=0.3)
        baseline_champ = pm.Normal("baseline_champ", mu=0.15, sigma=0.3)  # Prior: lower scoring
        
        home_adv_prem = pm.Normal("home_adv_prem", mu=0.25, sigma=0.2)
        home_adv_champ = pm.Normal("home_adv_champ", mu=0.20, sigma=0.2)  # Prior: slightly less home adv

        # Select parameters based on league
        baseline = baseline_prem * is_prem + baseline_champ * (1 - is_prem)
        home_adv = home_adv_prem * is_prem + home_adv_champ * (1 - is_prem)

        # Expected goals
        home_goals_mu = pm.math.exp(baseline + att_str[home_idx] + def_str[away_idx] + home_adv)
        away_goals_mu = pm.math.exp(baseline + att_str[away_idx] + def_str[home_idx])

        # Weighted likelihood
        weights = pm.ConstantData("weights", train_df["weight"].values)
        home_logp = pm.logp(pm.Poisson.dist(mu=home_goals_mu), home_goals_obs)
        away_logp = pm.logp(pm.Poisson.dist(mu=away_goals_mu), away_goals_obs)
        pm.Potential("weighted_home_goals", pm.math.sum(weights * home_logp))
        pm.Potential("weighted_away_goals", pm.math.sum(weights * away_logp))

        # Sample
        trace = pm.sample(trace=trace, tune=tune, cores=4, nuts_sampler='blackjax', 
                         return_inferencedata=True, progressbar=False)
    
    return model, trace


def convert_priors_dict_to_arrays(custom_priors, team_mapping, n_teams):
    """Convert custom priors dict to arrays for PyMC model"""
    
    # Initialize arrays with default values
    att_prior_mu = np.zeros(n_teams)
    att_prior_sigma = np.ones(n_teams)
    def_prior_mu = np.zeros(n_teams)
    def_prior_sigma = np.ones(n_teams)
    
    # Fill in custom priors where specified
    for team_name, priors in custom_priors.items():
        if team_name in team_mapping:
            idx = team_mapping[team_name]
            
            # Attack priors
            if 'attack_mu' in priors:
                att_prior_mu[idx] = priors['attack_mu']
            if 'attack_sigma' in priors:
                att_prior_sigma[idx] = priors['attack_sigma']
                
            # Defense priors  
            if 'defense_mu' in priors:
                def_prior_mu[idx] = priors['defense_mu']
            if 'defense_sigma' in priors:
                def_prior_sigma[idx] = priors['defense_sigma']
        else:
            print(f"Warning: Team '{team_name}' not found in team_mapping")
    
    return att_prior_mu, att_prior_sigma, def_prior_mu, def_prior_sigma


def convert_to_expected_goals_constrained(att_summary, def_summary, baseline, home_adv, teams):
    """Convert team strengths to expected goals vs league average opponent"""
    results = []
    
    # With sum-to-zero constraints, league average is simply 0
    att_league_avg = 0.0
    def_league_avg = 0.0
    
    for team in teams:
        team_att = att_summary.loc[team, 'mean']
        team_def = def_summary.loc[team, 'mean']
        
        # Expected goals FOR this team vs league average opponent
        goals_for = np.exp(baseline + team_att + def_league_avg)
        
        # Expected goals AGAINST this team vs league average opponent  
        goals_against = np.exp(baseline + att_league_avg + team_def)
        
        results.append({
            'Team': team,
            'Goals_For': goals_for,
            'Goals_Against': goals_against,
            'Goal_Diff': goals_for - goals_against
        })
    
    return pd.DataFrame(results)


def validate_model_predictions(trace, teams, train_df, n_simulations=1000):
    """
    Validate model predictions against observed data
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        Model trace
    teams : list
        List of team names
    train_df : pd.DataFrame
        Training data to compare against
    n_simulations : int
        Number of random fixture simulations
    """
    
    print("\n" + "=" * 60)
    print("MODEL PREDICTION VALIDATION")
    print("=" * 60)
    
    # Get observed statistics from training data
    train_df = train_df[
        (train_df["is_actual"] == True) & 
        (train_df["league_id"] == "Premier League") 
    ]
    observed_total_goals = train_df['home_goals'].sum() + train_df['away_goals'].sum()
    observed_matches = len(train_df)
    observed_goals_per_match = observed_total_goals / observed_matches
    observed_home_goals_per_match = train_df['home_goals'].mean()
    observed_away_goals_per_match = train_df['away_goals'].mean()
    
    print(f"\n1. OBSERVED DATA STATISTICS")
    print("-" * 40)
    print(f"Total matches: {observed_matches}")
    print(f"Total goals: {observed_total_goals}")
    print(f"Goals per match: {observed_goals_per_match:.3f}")
    print(f"Home goals per match: {observed_home_goals_per_match:.3f}")
    print(f"Away goals per match: {observed_away_goals_per_match:.3f}")
    
    # Get model parameters
    posterior = trace.posterior
    att_str = posterior.att_str.values
    def_str = posterior.def_str.values
    home_adv = posterior.home_adv.values
    baseline = posterior.baseline.values
    
    # Simulate random fixtures
    n_teams = len(teams)
    simulated_goals_per_match = []
    simulated_home_goals = []
    simulated_away_goals = []
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_simulations):
        # Pick random home and away teams
        home_team = np.random.randint(0, n_teams)
        away_team = np.random.randint(0, n_teams)
        while away_team == home_team:  # Ensure different teams
            away_team = np.random.randint(0, n_teams)
        
        # Pick random posterior samples
        chain_idx = np.random.randint(0, att_str.shape[0])
        draw_idx = np.random.randint(0, att_str.shape[1])
        
        # Calculate expected goals
        home_mu = np.exp(baseline[chain_idx, draw_idx] + 
                        att_str[chain_idx, draw_idx, home_team] + 
                        def_str[chain_idx, draw_idx, away_team] + 
                        home_adv[chain_idx, draw_idx])
        
        away_mu = np.exp(baseline[chain_idx, draw_idx] + 
                        att_str[chain_idx, draw_idx, away_team] + 
                        def_str[chain_idx, draw_idx, home_team])
        
        # Sample actual goals from Poisson
        home_goals = np.random.poisson(home_mu)
        away_goals = np.random.poisson(away_mu)
        
        simulated_home_goals.append(home_goals)
        simulated_away_goals.append(away_goals)
        simulated_goals_per_match.append(home_goals + away_goals)
    
    # Calculate simulated statistics
    sim_goals_per_match = np.mean(simulated_goals_per_match)
    sim_home_goals_per_match = np.mean(simulated_home_goals)
    sim_away_goals_per_match = np.mean(simulated_away_goals)
    sim_goals_std = np.std(simulated_goals_per_match)
    
    print(f"\n2. SIMULATED PREDICTIONS ({n_simulations:,} random fixtures)")
    print("-" * 40)
    print(f"Goals per match: {sim_goals_per_match:.3f} ± {sim_goals_std:.3f}")
    print(f"Home goals per match: {sim_home_goals_per_match:.3f}")
    print(f"Away goals per match: {sim_away_goals_per_match:.3f}")
    
    # Compare observed vs predicted
    goals_difference = sim_goals_per_match - observed_goals_per_match
    home_difference = sim_home_goals_per_match - observed_home_goals_per_match
    away_difference = sim_away_goals_per_match - observed_away_goals_per_match
    
    print(f"\n3. MODEL VALIDATION")
    print("-" * 40)
    print(f"Goals per match difference: {goals_difference:+.3f}")
    print(f"Home goals difference: {home_difference:+.3f}")
    print(f"Away goals difference: {away_difference:+.3f}")
    
    # Assessment
    print(f"\n4. ASSESSMENT")
    print("-" * 40)
    if abs(goals_difference) < 0.2:
        print("✓ EXCELLENT: Model predictions very close to observed data")
    elif abs(goals_difference) < 0.4:
        print("✓ GOOD: Model predictions reasonably close to observed data")
    elif abs(goals_difference) < 0.6:
        print("⚠ WARNING: Model predictions somewhat off from observed data")
    else:
        print("❌ POOR: Model predictions significantly different from observed data")
    
    if goals_difference > 0.4:
        print("  → Model predicting too many goals - consider adjusting baseline")
    elif goals_difference < -0.4:
        print("  → Model predicting too few goals - consider adjusting baseline")
    
    # Additional checks
    home_advantage_check = sim_home_goals_per_match - sim_away_goals_per_match
    observed_home_advantage = observed_home_goals_per_match - observed_away_goals_per_match
    
    print(f"\n5. HOME ADVANTAGE CHECK")
    print("-" * 40)
    print(f"Observed home advantage: {observed_home_advantage:.3f} goals")
    print(f"Predicted home advantage: {home_advantage_check:.3f} goals")
    print(f"Difference: {home_advantage_check - observed_home_advantage:+.3f}")
    
    return {
        'observed_goals_per_match': observed_goals_per_match,
        'predicted_goals_per_match': sim_goals_per_match,
        'goals_difference': goals_difference,
        'observed_home_advantage': observed_home_advantage,
        'predicted_home_advantage': home_advantage_check,
        'simulation_results': {
            'goals_per_match': simulated_goals_per_match,
            'home_goals': simulated_home_goals,
            'away_goals': simulated_away_goals
        }
    }


def analyze_model_results(trace, teams):
    """Comprehensive analysis of model results"""
    
    print("=" * 60)
    print("MODEL ANALYSIS RESULTS")
    print("=" * 60)
    
    # Summary statistics for all parameters
    print("\n1. PARAMETER SUMMARY STATISTICS")
    print("-" * 40)
    summary_stats = az.summary(trace)
    print(summary_stats)
    
    # Trace plots for diagnostics
    print("\n2. GENERATING TRACE PLOTS...")
    print("-" * 40)
    az.plot_trace(trace, var_names=[
        "att_str_raw", 
        "def_str_raw", 
        "att_str",     
        "def_str",            
        "home_adv",
        "baseline"
    ])
    
    # Get key parameter estimates
    home_adv_mean = az.summary(trace, var_names=["home_adv"])['mean'].iloc[0]
    baseline_mean = az.summary(trace, var_names=["baseline"])['mean'].iloc[0]
    
    print(f"\n3. KEY PARAMETERS")
    print("-" * 40)
    print(f"Home Advantage: {home_adv_mean:.4f}")
    print(f"Baseline Goals: {baseline_mean:.4f}")
    
    # Team strength rankings
    print("\n4. TEAM STRENGTH RANKINGS")
    print("-" * 40)
    
    # Attack rankings
    att_summary = az.summary(trace, var_names=["att_str"])
    att_summary.index = teams
    print("\nATTACK STRENGTH RANKINGS (higher is better):")
    att_rankings = att_summary[['mean', 'hdi_3%', 'hdi_97%']].sort_values("mean", ascending=False)
    print(att_rankings)
    
    # Defense rankings  
    def_summary = az.summary(trace, var_names=["def_str"])
    def_summary.index = teams
    print("\nDEFENSE STRENGTH RANKINGS (lower is better):")
    def_rankings = def_summary[['mean', 'hdi_3%', 'hdi_97%']].sort_values("mean", ascending=True)
    print(def_rankings)
    
    # Expected goals analysis
    print("\n5. EXPECTED GOALS VS LEAGUE AVERAGE")
    print("-" * 40)
    goals_df = convert_to_expected_goals_constrained(
        att_summary, def_summary, baseline_mean, home_adv_mean, teams
    )
    
    goals_table = goals_df[['Team', 'Goals_For', 'Goals_Against', 'Goal_Diff']].sort_values('Goal_Diff', ascending=False)
    print(goals_table.to_string(index=False, float_format='%.3f'))
    
    # Validation checks
    print(f"\n6. VALIDATION CHECKS")
    print("-" * 40)
    print(f"Total Goals For: {goals_df['Goals_For'].sum():.3f}")
    print(f"Total Goals Against: {goals_df['Goals_Against'].sum():.3f}")
    print(f"Difference (should be ~0): {goals_df['Goals_For'].sum() - goals_df['Goals_Against'].sum():.6f}")
    
    return {
        'summary': summary_stats,
        'attack_rankings': att_rankings,
        'defense_rankings': def_rankings,
        'expected_goals': goals_table,
        'home_advantage': home_adv_mean,
        'baseline': baseline_mean
    }


def run_full_analysis(train_df, teams, n_teams, season, 
                     team_mapping=None, trace_samples=5000, tune_samples=2500, 
                     model_version="v1", custom_priors=None):
    """
    Run complete model fitting and analysis with custom priors from dict
    
    Parameters:
    -----------
    custom_priors : dict, optional
        Custom team prior specifications as dictionary
        
    Returns:
    --------
    tuple: (model, trace, results)
    """
    
    # Extract year from season string if needed
    if isinstance(season, str):
        current_season = int(season[:4])
    else:
        current_season = season
    
    print(f"Season: {current_season}")
    print(f"Teams: {n_teams}")
    print(f"Model version: {model_version}")
    print(f"Samples: {trace_samples} (tune: {tune_samples})")
    
    if custom_priors:
        print(f"Custom priors specified for {len(custom_priors)} teams")
    
    print("\nBuilding and sampling model...")
    model, trace = build_and_sample_model(
        train_df=train_df, 
        n_teams=n_teams, 
        trace=trace_samples, 
        tune=tune_samples
    )
    
    print("\nAnalyzing results...")
    results = analyze_model_results(trace, teams)
    
    print("\nValidating model predictions...")
    validation_results = validate_model_predictions(trace, teams, train_df)
    results['validation'] = validation_results
    
    print(f"\nSaving trace for future use...")
    if team_mapping:
        team_names = list(team_mapping.keys())
        save_season_trace(trace, current_season, league, team_names, model_version)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return model, trace, results