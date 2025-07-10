import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

def add_calibrated_predictions_to_csv(input_file, output_file):
    """
    Add calibrated probability predictions to your CSV file
    
    Parameters:
    -----------
    input_file : str
        Path to your team_shots_calibration.csv
    output_file : str
        Path where to save the CSV with calibrated predictions
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file)

    #print(df.dtypes)
    
    # Create outcome column (1 if bet won, 0 if lost)
    df['won'] = np.where(
        df['outcome_name'] == 'Over',
        df['actual_shots'] > df['line'],
        df['actual_shots'] < df['line']
    ).astype(int)
    
    # Split data for training calibration (80/20 split)
    # We use the first 80% for training to maintain temporal order
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point].copy()
    
    print(f"Training calibration on first {len(train_df)} rows...")
    
    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(
        train_df['poisson_sim_prob'].values,
        train_df['won'].values
    )
    
    # Apply calibration to ALL rows
    print("Applying calibration to all predictions...")
    df['calibrated_prob'] = iso_reg.transform(df['poisson_sim_prob'].values)
    
    # Calculate edges using calibrated probabilities
    df['calibrated_edge'] = df['calibrated_prob'] - df['implied_prob']
    
    # Add betting recommendations
    MIN_EDGE = 0.05  # 5% minimum edge
    df['bet_calibrated'] = (df['calibrated_edge'] > MIN_EDGE).astype(int)
    
    # Calculate what the profit/loss would have been
    df['pnl_if_bet'] = np.where(
        df['won'] == 1,
        df['odds'] - 1,  # Profit if won
        -1  # Loss if lost
    )
    
    # Only count P&L for recommended bets
    df['calibrated_pnl'] = df['pnl_if_bet'] * df['bet_calibrated']
    
    # Add some summary stats
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"Average original probability: {df['poisson_sim_prob'].mean():.3f}")
    print(f"Average calibrated probability: {df['calibrated_prob'].mean():.3f}")
    print(f"Actual win rate: {df['won'].mean():.3f}")
    
    print(f"\nOriginal value bets: {(df['is_value_bet'] == True).sum()}")
    print(f"Calibrated bets (>{MIN_EDGE*100}% edge): {df['bet_calibrated'].sum()}")
    
    # Quick performance check
    original_bets = df[df['is_value_bet'] == True]
    calibrated_bets = df[df['bet_calibrated'] == 1]
    
    if len(original_bets) > 0:
        original_roi = (original_bets['pnl_if_bet'].sum() / len(original_bets)) * 100
        print(f"\nOriginal strategy ROI: {original_roi:.2f}%")
        print(f"Win rate on original bets: {original_bets['won'].mean()*100:.1f}%")
    
    if len(calibrated_bets) > 0:
        calibrated_roi = (calibrated_bets['pnl_if_bet'].sum() / len(calibrated_bets)) * 100
        print(f"Calibrated strategy ROI: {calibrated_roi:.2f}%")
        print(f"Win rate on calibrated bets: {calibrated_bets['won'].mean()*100:.1f}%")
    
    # Save to file
    print(f"\nSaving results to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("Done! New columns added:")
    print("- calibrated_prob: Calibrated probability after isotonic regression")
    print("- calibrated_edge: calibrated_prob - implied_prob")
    print("- bet_calibrated: 1 if calibrated_edge > 5%, 0 otherwise")
    print("- pnl_if_bet: Profit/loss for each bet (odds-1 if won, -1 if lost)")
    print("- calibrated_pnl: Actual P&L (0 if no bet recommended)")
    
    return df

# Simple example to verify calibration is working
def show_calibration_example(df, n_examples=5):
    """Show a few examples of how calibration changes predictions"""
    print("\n=== CALIBRATION EXAMPLES ===")
    print("Original Prob -> Calibrated Prob (Change)")
    
    # Show examples across different probability ranges
    for prob_threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        examples = df[
            (df['poisson_sim_prob'] > prob_threshold) & 
            (df['poisson_sim_prob'] < prob_threshold + 0.1)
        ].head(1)
        
        if len(examples) > 0:
            row = examples.iloc[0]
            change = row['calibrated_prob'] - row['poisson_sim_prob']
            print(f"{row['poisson_sim_prob']:.3f} -> {row['calibrated_prob']:.3f} ({change:+.3f})")

# Usage
if __name__ == "__main__":
    # Adjust these paths to your files
    input_file = r'C:\Users\Owner\dev\algobetting\algo\models\team_shots\team_shots_calibration.csv'
    output_file = 'team_shots_calibrated.csv'
    


    # Add calibrated predictions
    df = add_calibrated_predictions_to_csv(input_file, output_file)
    
    # Show some examples
    show_calibration_example(df)