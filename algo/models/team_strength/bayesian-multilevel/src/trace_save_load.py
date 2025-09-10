import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import pickle
import os
from pathlib import Path


def save_season_trace(trace, season_year, league, team_names=None, model_version="v1"):
    """Save trace with team name mapping for future use"""
    save_dir = Path(f"model_traces/{model_version}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the trace
    trace_path = save_dir / f"trace_{league}_{season_year}.nc"
    trace.to_netcdf(trace_path)
    
    # Save team name mapping if provided
    if team_names is not None:
        mapping_path = save_dir / f"team_mapping_{league}_{season_year}.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(team_names, f)
    
    # Save summary with readable names if possible
    summary_path = save_dir / f"summary_{league}_{season_year}.csv"
    summary = az.summary(trace)
    
    if team_names is not None:
        try:
            # Create readable parameter names for ALL indexed parameters
            readable_index = []
            for idx in summary.index:
                param_name = str(idx)
                
                # Check if this parameter has an index [X] pattern
                if '[' in param_name and ']' in param_name:
                    try:
                        # Extract the base parameter name and index
                        base_param = param_name.split('[')[0]
                        index_part = param_name.split('[')[1].split(']')[0]
                        
                        # Try to parse as integer index
                        try:
                            team_idx = int(index_part)
                            # Replace index with team name if valid
                            if 0 <= team_idx < len(team_names):
                                readable_index.append(f"{base_param}[{team_names[team_idx]}]")
                            else:
                                readable_index.append(param_name)  # Keep original if index out of bounds
                        except ValueError:
                            # Index is not a number (already a team name), keep as is
                            readable_index.append(param_name)
                            
                    except (ValueError, IndexError):
                        # Keep original if parsing fails
                        readable_index.append(param_name)
                else:
                    # No index - parameters like 'home_adv', 'baseline', etc.
                    readable_index.append(param_name)
            
            summary.index = readable_index
        except Exception as e:
            print(f"Warning: Could not create readable parameter names: {e}")
            # Just use the original summary if formatting fails
    
    summary.to_csv(summary_path)
    
    print(f"Saved trace for {season_year}")
    if team_names is None:
        print("Warning: Team names not saved - prior extraction from this trace will not work")
    
    return trace_path


def load_previous_season_trace(current_season, league, model_version="v1"):
    """Load both trace and team mapping from previous season"""
    save_dir = Path(f"model_traces/{model_version}")
    previous_year = current_season - 1
    
    trace_path = save_dir / f"trace_{league}_{previous_year}.nc"
    mapping_path = save_dir / f"team_mapping_{league}_{previous_year}.pkl"
    
    if trace_path.exists() and mapping_path.exists():
        try:
            trace = az.from_netcdf(trace_path)
            with open(mapping_path, 'rb') as f:
                team_names = pickle.load(f)
            print(f"Loaded trace and team mapping for {previous_year}")
            return trace, team_names
        except Exception as e:
            print(f"Failed to load data for {previous_year}: {e}")
            return None, None
    else:
        print(f"No complete data found for {previous_year}")
        return None, None


def extract_previous_season_priors(previous_trace, previous_team_names, 
                                 current_team_names, prior_strength=0.5):
    """
    Extract team strength priors using team names as stable identifiers
    
    Parameters:
    -----------
    previous_trace : arviz.InferenceData
        Trace from previous season
    previous_team_names : list
        Team names from previous season (in index order)
    current_team_names : list  
        Team names for current season (in index order)
    prior_strength : float
        Controls prior variance shrinkage
    
    Returns:
    --------
    dict with prior means and sigmas
    """
    if previous_trace is None or previous_team_names is None:
        return None
    
    # Get posterior means from previous season
    posterior = previous_trace.posterior
    prev_att_mean = posterior.att_str.mean(dim=['chain', 'draw']).values
    prev_def_mean = posterior.def_str.mean(dim=['chain', 'draw']).values
    
    # Create mapping from previous team names to their strength values
    prev_team_strengths = {}
    for i, team_name in enumerate(previous_team_names):
        prev_team_strengths[team_name] = {
            'att': prev_att_mean[i],
            'def': prev_def_mean[i]
        }
    
    # Initialize priors for current season
    n_teams = len(current_team_names)
    att_prior_mu = np.zeros(n_teams)
    def_prior_mu = np.zeros(n_teams)
    att_prior_sigma = np.ones(n_teams)  # Default for new teams
    def_prior_sigma = np.ones(n_teams)
    
    # Map previous strengths to current season using team names
    teams_with_history = 0
    new_teams = []
    
    for current_idx, team_name in enumerate(current_team_names):
        if team_name in prev_team_strengths:
            # Team existed last season - use their strength as prior
            att_prior_mu[current_idx] = prev_team_strengths[team_name]['att']
            def_prior_mu[current_idx] = prev_team_strengths[team_name]['def']
            att_prior_sigma[current_idx] = prior_strength
            def_prior_sigma[current_idx] = prior_strength
            teams_with_history += 1
        else:
            # New team - use default priors
            new_teams.append(team_name)
            # att_prior_mu[current_idx] = 0  # Already initialized
            # def_prior_mu[current_idx] = 0  # Already initialized
    
    print(f"Using priors from previous season for {teams_with_history}/{n_teams} teams")
    if new_teams:
        print(f"New teams (using default priors): {new_teams}")
    
    return {
        'att_prior_mu': att_prior_mu,
        'att_prior_sigma': att_prior_sigma,
        'def_prior_mu': def_prior_mu,
        'def_prior_sigma': def_prior_sigma,
        'teams_with_history': teams_with_history,
        'new_teams': new_teams
    }