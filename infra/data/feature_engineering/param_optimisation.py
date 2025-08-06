import pandas as pd
import sqlite3
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import poisson
from itertools import product
import time
from numba import jit

@jit(nopython=True)
def calculate_weights_vectorized(days_diff, red_cards, decay_rate=0.0077):
    """Vectorized weight calculation using numba for speed"""
    weights = np.exp(-days_diff * decay_rate)
    # Reduce weight for matches with red cards
    weights = np.where(red_cards == 1, weights * 0.3, weights)
    return weights

def apply_weighted_avg_vectorized(values, days_diff, red_cards, 
                                decay_rate=0.0077, time_window=180, 
                                min_games=5, recent_game_window=120,
                                red_card_penalty=0.3):
    """Optimized weighted average calculation"""
    # Create masks once
    valid_mask = ~pd.isna(values)
    time_window_mask = days_diff <= time_window
    recent_mask = days_diff <= recent_game_window
    
    # Combined mask
    combined_mask = valid_mask & time_window_mask
    
    # Early returns
    if not combined_mask.any():
        return np.nan
    if combined_mask.sum() < min_games:
        return np.nan
    if not (valid_mask & recent_mask).any():
        return np.nan
    
    # Filter data once
    valid_values = values[combined_mask]
    valid_days = days_diff[combined_mask]
    valid_red = red_cards[combined_mask]
    
    # Calculate weights (using the red_card_penalty parameter)
    weights = np.exp(-valid_days * decay_rate)
    weights = np.where(valid_red == 1, weights * red_card_penalty, weights)
    
    # Calculate weighted average
    try:
        valid_values_numeric = pd.to_numeric(valid_values, errors='coerce')
        if pd.isna(valid_values_numeric).all():
            return np.nan
        
        # Remove NaN values from conversion
        final_mask = ~pd.isna(valid_values_numeric)
        if not final_mask.any():
            return np.nan
            
        final_values = valid_values_numeric[final_mask]
        final_weights = weights[final_mask]
        
        return np.sum(final_weights * final_values) / np.sum(final_weights)
    except:
        return np.nan

def process_team_metrics_optimized(df, promoted_teams, decay_rate=0.0077, time_window=180, 
                                 min_games=5, recent_game_window=120, 
                                 red_card_penalty=0.3):
    """Optimized processing of team metrics"""
    
    # Clean and prepare data (same as original)
    df = df.drop_duplicates(subset=['match_url', 'team'])
    df['match_date'] = pd.to_datetime(df['match_date'])
    df['match_red'] = df["summary_cards_red"].astype(int) + df["opp_summary_cards_red"].astype(int)
    df['match_id'] = df['match_url']

    # Identify numeric columns (same logic as original)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['match_red', 'summary_cards_red', 'opp_summary_cards_red', 'is_home']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    team_stats = [col for col in numeric_cols if not col.startswith('opp_')]
    opp_stats = [col for col in numeric_cols if col.startswith('opp_')]

    print(f"Processing {len(team_stats)} team stats and {len(opp_stats)} opponent stats")

    # Sort data once for all teams
    df_sorted = df.sort_values(['team', 'match_date']).reset_index(drop=True)
    
    results = []
    
    # Group by team to process each team's matches
    for team_name, team_data in df_sorted.groupby('team'):
        team_data = team_data.reset_index(drop=True)
        n_matches = len(team_data)
        
        if n_matches <= 1:
            continue
            
        # Pre-compute match dates as numpy array for faster operations
        match_dates = team_data['match_date'].values
        match_red = team_data['match_red'].values
        
        # Process each match (starting from index 1, same as original)
        for current_idx in range(1, n_matches):
            current_match = team_data.iloc[current_idx]
            current_date = match_dates[current_idx]
            
            # Calculate days difference for all previous matches at once
            prev_dates = match_dates[:current_idx]
            prev_red = match_red[:current_idx]
            days_diff = (current_date - prev_dates) / np.timedelta64(1, 'D')
            
            metrics_dict = {
                'team': team_name,
                'match_id': current_match['match_id'],
                'opp_team': current_match['opp_team']
            }
            
            # Process team stats
            for stat in team_stats:
                prev_values = team_data[stat].iloc[:current_idx]
                
                weighted_avg = apply_weighted_avg_vectorized(
                    prev_values.values, days_diff, prev_red,
                    decay_rate=decay_rate, time_window=time_window,
                    min_games=min_games, recent_game_window=recent_game_window,
                    red_card_penalty=red_card_penalty
                )
                metrics_dict[f'team_rolling_{stat}'] = weighted_avg
            
            # Process opponent stats (defensive perspective)
            for stat in opp_stats:
                prev_values = team_data[stat].iloc[:current_idx]
                
                weighted_avg = apply_weighted_avg_vectorized(
                    prev_values.values, days_diff, prev_red,
                    decay_rate=decay_rate, time_window=time_window,
                    min_games=min_games, recent_game_window=recent_game_window,
                    red_card_penalty=red_card_penalty
                )
                
                clean_stat = stat.replace('opp_', '')
                metrics_dict[f'team_rolling_conceded_{clean_stat}'] = weighted_avg
            
            results.append(metrics_dict)
    
    # Convert to DataFrame once
    metrics_df = pd.DataFrame(results)
    
    # Merge operations (same as original but more efficient)
    final_df = df.merge(
        metrics_df,
        on=['team', 'match_id', 'opp_team'],
        how='left'
    )

    # Create opponent metrics
    opp_metrics_df = metrics_df.copy()
    opp_metrics_df = opp_metrics_df.rename(columns={
        'team': 'opp_team',
        'opp_team': 'team'
    })

    rolling_cols = [col for col in opp_metrics_df.columns if col.startswith('team_rolling_')]
    rename_dict = {col: f'opp_{col}' for col in rolling_cols}
    opp_metrics_df = opp_metrics_df.rename(columns=rename_dict)

    final_df = final_df.merge(
        opp_metrics_df,
        on=['team', 'match_id', 'opp_team'],
        how='left'
    )

    # Add the additional features AFTER rolling calculations (like in your original code)
    features_df = final_df.copy()
    features_df['gw'] = features_df.groupby(['team', 'season', 'division'])['match_date'].rank(method='dense').astype(int)
    features_df['is_early_season?'] = np.where(features_df['gw'] < 10, 1, 0)
    features_df['is_promoted?'] = features_df.merge(promoted_teams, on=['team', 'division', 'season'], how='left', indicator=True)['_merge'].eq('both').astype(int)

    # Select final columns (same as original)
    key_cols = ['match_url', 'match_date', 'season', 'division', 'team', 'opp_team', 
                'is_home', 'gw', 'is_promoted?', 'is_early_season?']
    rolling_cols = [col for col in features_df.columns if 'rolling' in col]
    feature_cols = key_cols + rolling_cols
    final_features_df = features_df[feature_cols].copy()
    
    return final_features_df

def poisson_log_likelihood(y_true, y_pred):
    """
    Calculate log-likelihood for Poisson distribution
    Higher values are better (less negative)
    """
    # Avoid log(0) and numerical issues
    y_pred = np.maximum(y_pred, 1e-10)
    
    # Calculate log-likelihood using scipy for robustness
    log_likelihood = np.sum(poisson.logpmf(y_true, y_pred))
    return log_likelihood

def poisson_deviance(y_true, y_pred):
    """
    Calculate Poisson deviance - lower is better
    """
    y_pred = np.maximum(y_pred, 1e-10)
    y_true_safe = np.maximum(y_true, 1e-10)
    
    deviance = 2 * np.sum(
        y_true * np.log(y_true_safe / y_pred) - (y_true - y_pred)
    )
    return deviance

def create_parameter_grid():
    """Create parameter combinations for time_window and decay_rate only"""
    param_grid = {
        'time_window': [90, 120, 150, 180, 210, 240, 300, 365],
        'decay_rate': [0.003, 0.005, 0.007, 0.0077, 0.01, 0.012, 0.015, 0.02]
    }
    
    combinations = list(product(*param_grid.values()))
    param_list = [dict(zip(param_grid.keys(), combo)) for combo in combinations]
    
    print(f"Generated {len(param_list)} parameter combinations")
    return param_list

def generate_sql_tables(df, promoted_teams, param_combinations, conn):
    """Generate SQL tables for all parameter combinations"""
    
    table_info = []
    
    for i, params in enumerate(param_combinations):
        tw = params['time_window']
        dr = params['decay_rate']
        
        # Create table name
        table_name = f"team_all_features_{tw}_{dr:.4f}".replace('.', '')
        
        print(f"Creating table {i+1}/{len(param_combinations)}: {table_name}")
        
        start_time = time.time()
        
        try:
            # Generate features with current parameters
            features_df = process_team_metrics_optimized(
                df, 
                promoted_teams,
                decay_rate=dr,
                time_window=tw,
                min_games=5,
                recent_game_window=120,
                red_card_penalty=0.3
            )
            
            # Write to SQL
            features_df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            processing_time = time.time() - start_time
            
            table_info.append({
                'table_name': table_name,
                'time_window': tw,
                'decay_rate': dr,
                'num_rows': len(features_df),
                'processing_time': processing_time,
                'success': True
            })
            
            print(f"  ✓ Created in {processing_time:.1f}s with {len(features_df)} rows")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            table_info.append({
                'table_name': table_name,
                'time_window': tw,
                'decay_rate': dr,
                'success': False,
                'error': str(e)
            })
    
    # Save table info
    table_df = pd.DataFrame(table_info)
    table_df.to_sql('parameter_tables_log', conn, if_exists='replace', index=False)
    
    return table_df

def evaluate_all_parameter_tables_log_likelihood(conn, cutoff_date='2024-08-01'):
    """Evaluate all parameter tables using log-likelihood as primary metric"""
    
    # Get list of all feature tables
    tables_query = """
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name LIKE 'team_all_features_%'
    ORDER BY name
    """
    
    tables_df = pd.read_sql_query(tables_query, conn)
    table_names = tables_df['name'].tolist()
    
    results = []
    
    for i, table_name in enumerate(table_names):
        print(f"Evaluating {i+1}/{len(table_names)}: {table_name}")
        
        # Parse parameters from table name
        parts = table_name.replace('team_all_features_', '').split('_')
        time_window = int(parts[0])
        decay_rate = float('0.' + parts[1]) if len(parts[1]) == 4 else float('0.0' + parts[1])
        
        try:
            # Your exact SQL query but with dynamic table name
            query = f"""
            SELECT DISTINCT
                f.match_date,
                f.team,
                f.is_home,
                f.gw,
                f.[is_promoted?],
                f.[is_early_season?],
                f.season,
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
                {table_name} f
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
            """
            
            df = pd.read_sql_query(query, conn)
            
            if len(df) == 0:
                print(f"  ✗ No data returned")
                continue
            
            # Preprocessing
            X = df.drop(columns=["team", "opp_team", "goals", "match_date"])
            X = pd.get_dummies(X, columns=["season"], drop_first=True)
            y = df["goals"]
            
            # Train/test split
            df['match_date'] = pd.to_datetime(df['match_date'])
            train_mask = df['match_date'] < cutoff_date
            test_mask = df['match_date'] >= cutoff_date
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            if len(X_test) == 0 or len(X_train) == 0:
                print(f"  ✗ Insufficient train/test data")
                continue
            
            # XGBoost setup
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            params = {
                'objective': 'count:poisson',
                'max_depth': 7,
                'eta': 0.1,
                'subsample': 0.7,
                'colsample_bytree': 0.8,
                'eval_metric': 'poisson-nloglik'
            }
            
            num_rounds = 100
            eval_list = [(dtrain, 'train'), (dtest, 'eval')]
            
            model = xgb.train(
                params, 
                dtrain, 
                num_rounds, 
                eval_list, 
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Make predictions and calculate ALL metrics
            preds = model.predict(dtest)
            
            # Traditional metrics (for comparison)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            
            # Poisson-specific metrics (PRIMARY OPTIMIZATION TARGETS)
            log_likelihood = poisson_log_likelihood(y_test, preds)
            mean_log_likelihood = log_likelihood / len(y_test)
            deviance = poisson_deviance(y_test, preds)
            
            # AIC for model comparison
            n_params = len(X.columns) + 1  # features + intercept
            aic = 2 * n_params - 2 * log_likelihood
            
            result = {
                'table_name': table_name,
                'time_window': time_window,
                'decay_rate': decay_rate,
                'log_likelihood': log_likelihood,
                'mean_log_likelihood': mean_log_likelihood,  # PRIMARY METRIC
                'poisson_deviance': deviance,
                'aic': aic,
                'rmse': rmse,  # Keep for comparison
                'mae': mae,    # Keep for comparison
                'train_size': len(X_train),
                'test_size': len(X_test),
                'num_features': len(X.columns),
                'success': True
            }
            
            print(f"  ✓ Log-Likelihood: {log_likelihood:.2f}, Mean LL: {mean_log_likelihood:.4f}, RMSE: {rmse:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            result = {
                'table_name': table_name,
                'time_window': time_window,
                'decay_rate': decay_rate,
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
    
    # Convert to DataFrame and save results
    results_df = pd.DataFrame(results)
    successful_results = results_df[results_df['success'] == True].copy()
    
    if len(successful_results) > 0:
        # Sort by HIGHEST log-likelihood (best model fit for Poisson)
        successful_results = successful_results.sort_values('log_likelihood', ascending=False)
        successful_results.to_sql('parameter_optimization_log_likelihood_results', conn, if_exists='replace', index=False)
        
        print(f"\n=== TOP 5 BEST PARAMETERS (by Log-Likelihood) ===")
        for i, row in successful_results.head().iterrows():
            print(f"{i+1}. TW:{row['time_window']}, DR:{row['decay_rate']:.4f} -> LL:{row['log_likelihood']:.2f}, Mean LL:{row['mean_log_likelihood']:.4f}, RMSE:{row['rmse']:.4f}")
        
        print(f"\n=== COMPARISON: TOP 5 BY RMSE ===")
        rmse_sorted = successful_results.sort_values('rmse')
        for i, row in rmse_sorted.head().iterrows():
            print(f"{i+1}. TW:{row['time_window']}, DR:{row['decay_rate']:.4f} -> RMSE:{row['rmse']:.4f}, LL:{row['log_likelihood']:.2f}")
    
    return results_df

# Usage example:
if __name__ == "__main__":
    # 1. Connect to your database
    conn = sqlite3.connect(r'C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db')

    # 2. Get the original raw match data
    original_df = pd.read_sql_query('''SELECT
                            team,
                            is_home,
                            match_date,
                            division,
                            season,
                            match_url,
                            summary_goals,
                            summary_pens_made,
                            summary_shots,
                            summary_shots_on_target,
                            summary_touches,
                            summary_xg,
                            summary_npxg,
                            possession_touches_att_pen_area,
                            keeper_psxg,
                            summary_cards_red,
                            opp_team,
                            opp_summary_goals,
                            opp_summary_pens_made,
                            opp_summary_shots,
                            opp_summary_shots_on_target,
                            opp_summary_touches,
                            opp_summary_xg,
                            opp_summary_npxg,
                            opp_possession_touches_att_pen_area,
                            opp_keeper_psxg,
                            opp_summary_cards_red            
                           FROM fbref_match_all_columns
                           WHERE division IN ('Premier League', 'Championship')''', conn)

    # 3. Load promoted teams
    promoted_teams = pd.read_csv("infra/data/feature_engineering/promoted_teams.csv")

    # 4. Generate parameter combinations
    param_combos = create_parameter_grid()

    # 5. Generate all SQL tables (if not already done)
    # table_info = generate_sql_tables(original_df, promoted_teams, param_combos, conn)

    # 6. Evaluate all tables using LOG-LIKELIHOOD as primary metric
    results = evaluate_all_parameter_tables_log_likelihood(conn)

    # 7. Get best parameters (by log-likelihood, not RMSE)
    successful_results = results[results['success'] == True]
    if len(successful_results) > 0:
        best_result = successful_results.sort_values('log_likelihood', ascending=False).iloc[0]
        print(f"\nBest parameters by LOG-LIKELIHOOD: TW={best_result['time_window']}, DR={best_result['decay_rate']}")
        print(f"Log-Likelihood: {best_result['log_likelihood']:.2f}")
        print(f"Mean Log-Likelihood: {best_result['mean_log_likelihood']:.4f}")
        print(f"RMSE: {best_result['rmse']:.4f}")

    conn.close()