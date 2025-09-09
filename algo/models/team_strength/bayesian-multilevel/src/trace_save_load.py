import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import pickle
import os
from pathlib import Path


def save_season_trace(trace, season_year, league, model_version="v1"):
    """Save trace for future use"""
    save_dir = Path(f"model_traces/{model_version}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ArviZ NetCDF format (recommended - preserves all metadata)
    trace_path = save_dir / f"trace_{league}_{season_year}.nc"
    trace.to_netcdf(trace_path)
    
    # Also save a summary for quick reference
    summary_path = save_dir / f"summary_{league}_{season_year}.csv"
    az.summary(trace).to_csv(summary_path)
    
    print(f"Saved trace for {season_year} to {trace_path}")
    return trace_path


def load_previous_season_trace(current_season, league, model_version="v1"):
    """Load trace from previous season only"""
    save_dir = Path(f"model_traces/{model_version}")
    previous_year = current_season - 1
    trace_path = save_dir / f"trace_{league}_{previous_year}.nc"
    
    if trace_path.exists():
        try:
            trace = az.from_netcdf(trace_path)
            print(f"Loaded trace for {previous_year}")
            return trace
        except Exception as e:
            print(f"Failed to load trace for {previous_year}: {e}")
            return None
    else:
        print(f"No trace found for {previous_year}")
        return None


def extract_previous_season_priors(previous_trace, team_mapping, prior_strength=0.5):
    """
    Extract team strength priors from previous season
    
    Parameters:
    -----------
    previous_trace : arviz.InferenceData
        Trace from previous season
    team_mapping : dict
        Mapping from team names to current season indices: {team_name: current_idx}
    prior_strength : float
        Controls how much to shrink the prior variance (0.5 = moderate shrinkage)
    
    Returns:
    --------
    dict with prior means and sigmas for teams that existed last season
    """
    if previous_trace is None:
        return None
    
    # Get posterior means from previous season
    posterior = previous_trace.posterior
    prev_att_mean = posterior.att_str.mean(dim=['chain', 'draw']).values
    prev_def_mean = posterior.def_str.mean(dim=['chain', 'draw']).values
    
    # Initialize priors for current season
    n_teams = len(team_mapping)
    att_prior_mu = np.zeros(n_teams)
    def_prior_mu = np.zeros(n_teams)
    att_prior_sigma = np.ones(n_teams)  # Default sigma of 1 for new teams
    def_prior_sigma = np.ones(n_teams)
    
    # Map previous season strengths to current season indices
    # This assumes you have team names or some way to map between seasons
    for team_name, current_idx in team_mapping.items():
        # You'll need to implement the logic to get previous_idx from team_name
        # For now, assuming team indices are consistent or you have a mapping
        try:
            # If you have consistent team indices between seasons:
            if current_idx < len(prev_att_mean):
                att_prior_mu[current_idx] = prev_att_mean[current_idx]
                def_prior_mu[current_idx] = prev_def_mean[current_idx]
                att_prior_sigma[current_idx] = prior_strength
                def_prior_sigma[current_idx] = prior_strength
        except:
            # Team not found in previous season - use defaults
            pass
    
    return {
        'att_prior_mu': att_prior_mu,
        'att_prior_sigma': att_prior_sigma,
        'def_prior_mu': def_prior_mu,
        'def_prior_sigma': def_prior_sigma
    }