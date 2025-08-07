# %%
import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna
import os
import json 


conn = sqlite3.connect(r'C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db')

df = pd.read_sql_query("""
    SELECT DISTINCT
        f.match_date,
        f.team,
        f.is_home,
        f.gw,
        f.[is_promoted?],
        f.[is_early_season?],
        f.season,
        f.division,
        team_elo.elo as team_elo,
        f.team_rolling_summary_goals,
        f.team_rolling_summary_pens_made,
        f.team_rolling_summary_shots,
        f.team_rolling_summary_shots_on_target,
        f.team_rolling_summary_touches,
        f.team_rolling_summary_xg,
        f.team_rolling_summary_npxg,
        f.team_rolling_possession_touches_att_pen_area,
        f.team_rolling_keeper_psxg,
        f.team_rolling_conceded_summary_goals,
        f.team_rolling_conceded_summary_pens_made,
        f.team_rolling_conceded_summary_shots,
        f.team_rolling_conceded_summary_shots_on_target,
        f.team_rolling_conceded_summary_touches,
        f.team_rolling_conceded_summary_xg,
        f.team_rolling_conceded_summary_npxg,
        f.team_rolling_conceded_possession_touches_att_pen_area,
        f.team_rolling_conceded_keeper_psxg,
        f.opp_team,
        opp_elo.elo as opp_team_elo,
        f.opp_team_rolling_summary_goals,
        f.opp_team_rolling_summary_pens_made,
        f.opp_team_rolling_summary_shots,
        f.opp_team_rolling_summary_shots_on_target,
        f.opp_team_rolling_summary_touches,
        f.opp_team_rolling_summary_xg,
        f.opp_team_rolling_summary_npxg,
        f.opp_team_rolling_possession_touches_att_pen_area,
        f.opp_team_rolling_keeper_psxg,
        f.opp_team_rolling_conceded_summary_goals,
        f.opp_team_rolling_conceded_summary_pens_made,
        f.opp_team_rolling_conceded_summary_shots,
        f.opp_team_rolling_conceded_summary_shots_on_target,
        f.opp_team_rolling_conceded_summary_touches,
        f.opp_team_rolling_conceded_summary_xg,
        f.opp_team_rolling_conceded_summary_npxg,
        f.opp_team_rolling_conceded_possession_touches_att_pen_area,
        f.opp_team_rolling_conceded_keeper_psxg,
        ms.summary_goals as goals
    FROM 
        team_all_features_90_00200 f
    JOIN
        fbref_match_all_columns ms
            ON ms.match_url = f.match_url
            AND ms.team = f.team
    JOIN
        clubelo_features team_elo
            ON team_elo.fbref_team = f.team
                AND DATE(f.match_date) BETWEEN team_elo.start_date AND team_elo.end_date
    JOIN
        clubelo_features opp_elo
            ON opp_elo.fbref_team = f.opp_team
                AND DATE(f.match_date) BETWEEN opp_elo.start_date AND opp_elo.end_date
    WHERE 
        (f.team_rolling_summary_goals IS NOT NULL AND f.opp_team_rolling_summary_goals IS NOT NULL)
        AND f.division = 'Premier League'
                       """, conn)

df

# %%
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=["team", "opp_team", "goals", "match_date"])
X = pd.get_dummies(X, columns=["season", "division"], drop_first=True)

y = df["goals"]

# Use matches after a certain date as test
cutoff_date = '2024-08-01'
df['match_date'] = pd.to_datetime(df['match_date'])

train_mask = df['match_date'] < cutoff_date
test_mask = df['match_date'] >= cutoff_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
from sklearn.linear_model import PoissonRegressor
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fit Poisson regression
print("\nTraining Poisson Regression baseline...")
poisson_reg = PoissonRegressor(
    alpha=1.0,          # Regularization strength
    max_iter=1000,      # Increase if convergence issues
    fit_intercept=True
)

poisson_reg.fit(X_train_scaled, y_train)

# Make predictions
preds_poisson = poisson_reg.predict(X_test_scaled)

# Calculate traditional metrics
rmse_poisson = np.sqrt(mean_squared_error(y_test, preds_poisson))
mae_poisson = mean_absolute_error(y_test, preds_poisson)

# Calculate log-likelihood
def poisson_log_likelihood(y_true, y_pred):
    """
    Calculate log-likelihood for Poisson distribution
    y_true: actual goal counts
    y_pred: predicted lambda (rate parameter)
    """
    # Avoid log(0) by adding small epsilon
    y_pred = np.maximum(y_pred, 1e-10)
    
    # Poisson log-likelihood: y*log(λ) - λ - log(y!)
    log_likelihood = np.sum(
        y_true * np.log(y_pred) - y_pred - np.log(np.maximum(1, np.arange(1, len(y_true) + 1)))
    )
    return log_likelihood

def poisson_log_likelihood_scipy(y_true, y_pred):
    """
    Alternative using scipy.stats - more robust
    """
    y_pred = np.maximum(y_pred, 1e-10)
    return np.sum(poisson.logpmf(y_true, y_pred))

# Calculate Poisson deviance
def poisson_deviance(y_true, y_pred):
    """
    Poisson deviance - lower is better
    """
    y_pred = np.maximum(y_pred, 1e-10)
    y_true_safe = np.maximum(y_true, 1e-10)
    
    deviance = 2 * np.sum(
        y_true * np.log(y_true_safe / y_pred) - (y_true - y_pred)
    )
    return deviance

# Calculate all metrics
log_likelihood = poisson_log_likelihood_scipy(y_test, preds_poisson)
deviance = poisson_deviance(y_test, preds_poisson)

# Mean log-likelihood (for interpretability)
mean_log_likelihood = log_likelihood / len(y_test)

print(f"\n=== POISSON REGRESSION BASELINE ===")
print(f"RMSE: {rmse_poisson:.4f}")
print(f"MAE: {mae_poisson:.4f}")
print(f"Log-likelihood: {log_likelihood:.2f}")
print(f"Mean log-likelihood: {mean_log_likelihood:.4f}")
print(f"Poisson deviance: {deviance:.2f}")
print(f"Mean prediction: {preds_poisson.mean():.2f}")
print(f"Mean actual: {y_test.mean():.2f}")

# Additional diagnostic: AIC (Akaike Information Criterion)
n_params = X_train_scaled.shape[1] + 1  # features + intercept
aic = 2 * n_params - 2 * log_likelihood
print(f"AIC: {aic:.2f}")

# Calibration check - compare predicted vs observed rates in bins
def calibration_analysis(y_true, y_pred, n_bins=10):
    """
    Check if predicted rates match observed rates across different ranges
    """
    # Create bins based on predicted values
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_pred, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    print(f"\n=== CALIBRATION ANALYSIS ===")
    print("Bin | Predicted | Observed | Count")
    print("-" * 35)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            pred_mean = y_pred[mask].mean()
            obs_mean = y_true[mask].mean()
            count = np.sum(mask)
            print(f"{i+1:2d}  | {pred_mean:8.3f} | {obs_mean:7.3f} | {count:4d}")

# Run calibration analysis
calibration_analysis(y_test, preds_poisson)

# %%
# ===== TEAM ATTACK AND DEFENSE STRENGTH CALCULATION =====
print("\n" + "="*60)
print("CALCULATING TEAM ATTACK AND DEFENSE STRENGTHS")
print("="*60)

def calculate_team_strengths(df, model, scaler, X_train_cols, original_df):
    """
    Calculate attack and defense strengths by comparing each team 
    against a median team for both home and away scenarios.
    """
    
    # Get the most recent data for each team
    df_sorted = original_df.sort_values(['team', 'match_date'])
    latest_team_data = df_sorted.groupby('team').last().reset_index()
    
    print(f"Calculating strengths for {len(latest_team_data)} teams")
    print(f"Using most recent data from: {latest_team_data['match_date'].min()} to {latest_team_data['match_date'].max()}")
    
    # Calculate median values for original numeric features (before dummies)
    original_numeric_features = [col for col in original_df.columns if col not in ['team', 'opp_team', 'match_date', 'goals']]
    median_features = original_df[original_numeric_features].median()
    
    # Also calculate median values for dummy variables from training data
    dummy_medians = {}
    for col in X_train_cols:
        if col.startswith('season_') or col.startswith('division_'):
            # For dummy variables, use the mode (most common value) from training data
            dummy_medians[col] = df[col].mode().iloc[0] if col in df.columns else 0
    
    print(f"\nMedian team stats (key features):")
    key_features = ['team_elo', 'team_rolling_summary_xg', 'team_rolling_summary_goals', 
                   'team_rolling_conceded_summary_goals', 'team_rolling_conceded_summary_xg']
    for feat in key_features:
        if feat in median_features:
            print(f"  {feat}: {median_features[feat]:.3f}")
    
    team_strengths = []
    
    for _, team_row in latest_team_data.iterrows():
        team_name = team_row['team']
        
        # === ATTACK STRENGTH CALCULATION ===
        # Team attacking vs median defense (both home and away)
        
        # Home attack scenario: Team at home vs median team away
        home_attack_data = team_row.copy()
        home_attack_data['is_home'] = 1
        home_attack_data['opp_team'] = 'median_team'
        
        # Set opponent features to median values
        for feat in original_numeric_features:
            if feat.startswith('opp_team_'):
                base_feat = feat.replace('opp_team_', 'team_')
                if base_feat in median_features:
                    home_attack_data[feat] = median_features[base_feat]
            elif feat.startswith('team_elo'):
                home_attack_data['opp_team_elo'] = median_features['team_elo']
        
        # Away attack scenario: Team away vs median team at home  
        away_attack_data = team_row.copy()
        away_attack_data['is_home'] = 0
        away_attack_data['opp_team'] = 'median_team'
        
        # Set opponent features to median values
        for feat in original_numeric_features:
            if feat.startswith('opp_team_'):
                base_feat = feat.replace('opp_team_', 'team_')
                if base_feat in median_features:
                    away_attack_data[feat] = median_features[base_feat]
            elif feat.startswith('team_elo'):
                away_attack_data['opp_team_elo'] = median_features['team_elo']
        
        # === DEFENSE STRENGTH CALCULATION ===
        # Median team attacking vs this team defending
        
        # Home defense scenario: Median team away vs this team at home
        home_defense_data = team_row.copy()
        home_defense_data['is_home'] = 1
        home_defense_data['team'] = 'median_team'
        home_defense_data['opp_team'] = team_name
        
        # Swap team and opponent features
        temp_data = home_defense_data.copy()
        for feat in original_numeric_features:
            if feat.startswith('team_') and not feat.startswith('team_rolling_conceded'):
                opp_feat = feat.replace('team_', 'opp_team_')
                if opp_feat in temp_data:
                    # Set median team features
                    base_feat = feat
                    if base_feat in median_features:
                        home_defense_data[feat] = median_features[base_feat]
                    # Move this team's features to opponent position
                    home_defense_data[opp_feat] = temp_data[feat]
            elif feat.startswith('opp_team_') and not feat.startswith('opp_team_rolling_conceded'):
                team_feat = feat.replace('opp_team_', 'team_')
                if team_feat in temp_data:
                    # This team's features go to opponent position (already done above)
                    pass
        
        # Set ELO values correctly
        home_defense_data['team_elo'] = median_features['team_elo']
        home_defense_data['opp_team_elo'] = team_row['team_elo']
        
        # Away defense scenario: Median team at home vs this team away
        away_defense_data = home_defense_data.copy()
        away_defense_data['is_home'] = 0
        
        # Prepare data for prediction
        scenarios = {
            'home_attack': home_attack_data,
            'away_attack': away_attack_data,
            'home_defense': home_defense_data,
            'away_defense': away_defense_data
        }
        
        predictions = {}
        
        for scenario_name, scenario_data in scenarios.items():
            # Prepare features same way as training data
            scenario_features = scenario_data.drop(['team', 'opp_team', 'goals', 'match_date'] if 'goals' in scenario_data else ['team', 'opp_team', 'match_date'])
            
            # Handle categorical variables - create dummies
            scenario_features = pd.get_dummies(scenario_features, columns=["season", "division"], drop_first=True)
            
            # Align columns with training data - add missing dummy columns
            for col in X_train_cols:
                if col not in scenario_features.columns:
                    if col in dummy_medians:
                        scenario_features[col] = dummy_medians[col]
                    else:
                        scenario_features[col] = 0
            
            # Select only the columns used in training and in the right order
            scenario_features = scenario_features.reindex(columns=X_train_cols, fill_value=0)
            
            # Scale and predict
            scenario_scaled = scaler.transform(scenario_features.values.reshape(1, -1))
            pred = model.predict(scenario_scaled)[0]
            predictions[scenario_name] = pred
        
        # Calculate strength metrics
        home_attack_strength = predictions['home_attack']
        away_attack_strength = predictions['away_attack'] 
        avg_attack_strength = (home_attack_strength + away_attack_strength) / 2
        
        # For defense, we want to know how many goals the median team scores against this team
        # Lower goals conceded = better defense
        home_defense_conceded = predictions['home_defense']
        away_defense_conceded = predictions['away_defense']
        avg_defense_conceded = (home_defense_conceded + away_defense_conceded) / 2
        
        # Defense strength: inverse of goals conceded (higher = better defense)
        league_avg_goals = y.mean()  # Use overall average as baseline
        home_defense_strength = league_avg_goals / max(home_defense_conceded, 0.1)
        away_defense_strength = league_avg_goals / max(away_defense_conceded, 0.1)
        avg_defense_strength = (home_defense_strength + away_defense_strength) / 2
        
        team_strengths.append({
            'team': team_name,
            'home_attack_strength': home_attack_strength,
            'away_attack_strength': away_attack_strength,
            'avg_attack_strength': avg_attack_strength,
            'home_defense_conceded': home_defense_conceded,
            'away_defense_conceded': away_defense_conceded,
            'avg_defense_conceded': avg_defense_conceded,
            'home_defense_strength': home_defense_strength,
            'away_defense_strength': away_defense_strength,
            'avg_defense_strength': avg_defense_strength,
            'current_elo': team_row['team_elo'],
            'latest_match_date': team_row['match_date']
        })
    
    return pd.DataFrame(team_strengths)

# Calculate team strengths
print("\nCalculating team strengths using trained Poisson model...")
team_strength_df = calculate_team_strengths(X, poisson_reg, scaler, X_train.columns.tolist(), df)

# Sort by average attack strength
team_strength_df = team_strength_df.sort_values('avg_attack_strength', ascending=False)

print(f"\n=== TEAM ATTACK RANKINGS ===")
print("Rank | Team                | Avg Attack | Home Attack | Away Attack | ELO")
print("-" * 75)
for i, (_, row) in enumerate(team_strength_df.iterrows()):
    print(f"{i+1:4d} | {row['team']:18s} | {row['avg_attack_strength']:10.3f} | {row['home_attack_strength']:11.3f} | {row['away_attack_strength']:11.3f} | {row['current_elo']:4.0f}")

# Sort by defense strength (higher = better)
team_strength_df_def = team_strength_df.sort_values('avg_defense_strength', ascending=False)

print(f"\n=== TEAM DEFENSE RANKINGS ===")
print("Rank | Team                | Avg Defense | Home Defense | Away Defense | Avg Conceded")
print("-" * 80)
for i, (_, row) in enumerate(team_strength_df_def.iterrows()):
    print(f"{i+1:4d} | {row['team']:18s} | {row['avg_defense_strength']:11.3f} | {row['home_defense_strength']:12.3f} | {row['away_defense_strength']:12.3f} | {row['avg_defense_conceded']:12.3f}")

# Overall team strength (combination of attack and defense)
team_strength_df['overall_strength'] = (
    (team_strength_df['avg_attack_strength'] / team_strength_df['avg_attack_strength'].mean()) + 
    (team_strength_df['avg_defense_strength'] / team_strength_df['avg_defense_strength'].mean())
) / 2

team_strength_df_overall = team_strength_df.sort_values('overall_strength', ascending=False)

print(f"\n=== OVERALL TEAM STRENGTH RANKINGS ===")
print("Rank | Team                | Overall | Attack | Defense | ELO")
print("-" * 65)
for i, (_, row) in enumerate(team_strength_df_overall.iterrows()):
    print(f"{i+1:4d} | {row['team']:18s} | {row['overall_strength']:7.3f} | {row['avg_attack_strength']:6.3f} | {row['avg_defense_strength']:7.3f} | {row['current_elo']:4.0f}")

# Compare with ELO rankings
print(f"\n=== COMPARISON: MODEL STRENGTH vs ELO ===")
elo_rank_df = team_strength_df.sort_values('current_elo', ascending=False).reset_index(drop=True)
model_rank_df = team_strength_df.sort_values('overall_strength', ascending=False).reset_index(drop=True)

print("Team                | Model Rank | ELO Rank | Difference | Model Score | ELO")
print("-" * 80)
for i, row in model_rank_df.iterrows():
    team = row['team']
    model_rank = i + 1
    elo_rank = elo_rank_df[elo_rank_df['team'] == team].index[0] + 1
    rank_diff = elo_rank - model_rank
    print(f"{team:18s} | {model_rank:10d} | {elo_rank:8d} | {rank_diff:+10d} | {row['overall_strength']:11.3f} | {row['current_elo']:4.0f}")

# Save team strengths for future use
team_strength_df.to_csv('team_strengths.csv', index=False)
print(f"\nTeam strengths saved to 'team_strengths.csv'")

print(f"\n=== SUMMARY STATISTICS ===")
print(f"League average goals per game: {y.mean():.3f}")
print(f"Attack strength range: {team_strength_df['avg_attack_strength'].min():.3f} to {team_strength_df['avg_attack_strength'].max():.3f}")
print(f"Defense conceded range: {team_strength_df['avg_defense_conceded'].min():.3f} to {team_strength_df['avg_defense_conceded'].max():.3f}")
print(f"Defense strength range: {team_strength_df['avg_defense_strength'].min():.3f} to {team_strength_df['avg_defense_strength'].max():.3f}")

# %%