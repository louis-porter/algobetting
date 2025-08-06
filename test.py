import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import poisson
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sqlite3

# ================================
# STEP 1: LOAD AND AGGREGATE TEAM DATA
# ================================

def load_and_aggregate_team_data():
    """Load match data and calculate team strength metrics"""
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
            is_home,
            match_date as date
        FROM fbref_match_all_columns
        WHERE division = 'Premier League'
            AND season = '2024-2025'
            AND summary_xg IS NOT NULL
            AND opp_summary_xg IS NOT NULL

    """, conn)
    conn.close()
    
    print(f"📊 Loaded {len(df)} individual match records")
    
    # Calculate team performance metrics over time
    team_stats = []
    
    for team in df['team'].unique():
        team_matches = df[df['team'] == team].sort_values('date')
        
        if len(team_matches) < 5:  # Skip teams with too few matches
            continue
            
        # Calculate rolling averages (last 10 games)
        window = min(10, len(team_matches))
        
        recent_stats = {
            'team': team,
            'matches_played': len(team_matches),
            'avg_goals_for': team_matches['goals_for'].rolling(window).mean().iloc[-1],
            'avg_goals_against': team_matches['goals_against'].rolling(window).mean().iloc[-1],
            'avg_xg_for': team_matches['xg_for'].rolling(window).mean().iloc[-1],
            'avg_xg_against': team_matches['xg_against'].rolling(window).mean().iloc[-1],
            'home_games': len(team_matches[team_matches['is_home'] == 1]),
            'away_games': len(team_matches[team_matches['is_home'] == 0]),
        }
        
        team_stats.append(recent_stats)
    
    team_df = pd.DataFrame(team_stats)
    print(f"✅ Calculated stats for {len(team_df)} teams")
    
    return team_df, df

# ================================
# STEP 2: NEURAL NETWORK FOR TEAM STRENGTHS
# ================================

class TeamStrengthModel(nn.Module):
    """
    This network takes team performance metrics and outputs:
    - Attack strength 
    - Defense strength
    That work well in Dixon-Coles style predictions
    """
    
    def __init__(self, n_features):
        super().__init__()
        
        # Shared feature processing
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        
        # Separate heads for attack and defense
        self.attack_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        self.defense_head = nn.Sequential(
            nn.Linear(16, 8), 
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # Home advantage parameter
        self.home_advantage = nn.Parameter(torch.tensor(0.3))
    
    def get_team_strengths(self, team_features):
        """Get attack and defense strengths for teams"""
        encoded = self.feature_encoder(team_features)
        
        # Use softplus to ensure positive values
        attack = torch.nn.functional.softplus(self.attack_head(encoded))
        defense = torch.nn.functional.softplus(self.defense_head(encoded))
        
        return attack.squeeze(), defense.squeeze()
    
    def forward(self, home_features, away_features):
        """Predict match outcome using team strengths"""
        home_attack, home_defense = self.get_team_strengths(home_features)
        away_attack, away_defense = self.get_team_strengths(away_features)
        
        # Dixon-Coles style calculation
        home_lambda = home_attack * away_defense * torch.exp(self.home_advantage)
        away_lambda = away_attack * home_defense
        
        return home_lambda, away_lambda

# ================================
# STEP 3: PREPARE TRAINING DATA
# ================================

def prepare_training_data(team_df, match_df):
    """Create training pairs from team stats and match results"""
    
    # Create team lookup
    team_to_idx = {team: idx for idx, team in enumerate(team_df['team'])}
    
    # Features for strength calculation
    feature_cols = ['avg_goals_for', 'avg_goals_against', 'avg_xg_for', 'avg_xg_against']
    team_features = team_df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    team_features_scaled = scaler.fit_transform(team_features)
    
    # Create training examples from recent matches
    training_data = []
    
    for _, match in match_df.iterrows():
        home_team = match['team'] if match['is_home'] == 1 else match['opp_team']
        away_team = match['opp_team'] if match['is_home'] == 1 else match['team']
        
        if home_team in team_to_idx and away_team in team_to_idx:
            home_idx = team_to_idx[home_team]
            away_idx = team_to_idx[away_team]
            
            home_goals = match['goals_for'] if match['is_home'] == 1 else match['goals_against']
            away_goals = match['goals_against'] if match['is_home'] == 1 else match['goals_for']
            
            training_data.append({
                'home_idx': home_idx,
                'away_idx': away_idx, 
                'home_goals': home_goals,
                'away_goals': away_goals,
                'home_team': home_team,
                'away_team': away_team
            })
    
    train_df = pd.DataFrame(training_data)
    
    print(f"🎯 Created {len(train_df)} training examples")
    
    return train_df, team_features_scaled, scaler, team_to_idx

# ================================
# STEP 4: LOSS FUNCTION
# ================================

def poisson_loss(predicted_lambda, actual_goals):
    """Poisson negative log-likelihood loss"""
    # Clamp lambda to avoid numerical issues
    predicted_lambda = torch.clamp(predicted_lambda, 0.01, 10.0)
    return -torch.distributions.Poisson(predicted_lambda).log_prob(actual_goals).mean()

# ================================
# STEP 5: TRAINING
# ================================

def train_team_strength_model():
    """Train the model to learn team strengths"""
    
    print("📊 Loading and processing team data...")
    team_df, match_df = load_and_aggregate_team_data()
    
    print("🔄 Preparing training data...")
    train_df, team_features, scaler, team_to_idx = prepare_training_data(team_df, match_df)
    
    if len(train_df) < 10:
        print("❌ Not enough training data!")
        return None, None, None, None
    
    # Convert to tensors
    team_features_tensor = torch.FloatTensor(team_features)
    
    # Training data
    home_indices = torch.LongTensor(train_df['home_idx'].values)
    away_indices = torch.LongTensor(train_df['away_idx'].values)
    home_goals = torch.FloatTensor(train_df['home_goals'].values)
    away_goals = torch.FloatTensor(train_df['away_goals'].values)
    
    # Create model
    model = TeamStrengthModel(n_features=team_features.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"\n🧠 Training model on {len(train_df)} matches...")
    
    # Training loop
    for epoch in range(500):
        optimizer.zero_grad()
        
        # Get team features for this batch
        home_features = team_features_tensor[home_indices]
        away_features = team_features_tensor[away_indices]
        
        # Predict match outcomes
        pred_home_goals, pred_away_goals = model(home_features, away_features)
        
        # Calculate loss
        loss = poisson_loss(pred_home_goals, home_goals) + poisson_loss(pred_away_goals, away_goals)
        
        # Update model
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print(f"\n✅ Training complete! Final loss: {loss.item():.4f}")
    print(f"🏠 Learned home advantage: {torch.exp(model.home_advantage).item():.3f}")
    
    return model, team_df, scaler, team_to_idx

# ================================
# STEP 6: EXTRACT AND DISPLAY TEAM STRENGTHS
# ================================

def show_team_strengths(model, team_df, scaler):
    """Display the learned team strengths"""
    
    if model is None:
        return
        
    model.eval()
    
    # Get features
    feature_cols = ['avg_goals_for', 'avg_goals_against', 'avg_xg_for', 'avg_xg_against']
    team_features = scaler.transform(team_df[feature_cols].values)
    team_features_tensor = torch.FloatTensor(team_features)
    
    # Get strengths
    with torch.no_grad():
        attack_strengths, defense_strengths = model.get_team_strengths(team_features_tensor)
    
    # Create results dataframe
    results = team_df[['team']].copy()
    results['attack_strength'] = attack_strengths.numpy()
    results['defense_strength'] = defense_strengths.numpy()
    
    # Sort by attack strength
    results_attack = results.sort_values('attack_strength', ascending=False)
    results_defense = results.sort_values('defense_strength', ascending=False)
    
    print("\n⚔️ ATTACK STRENGTHS (Highest = Best Attack)")
    print("=" * 50)
    for i, (_, row) in enumerate(results_attack.head(20).iterrows()):
        print(f"{i+1:2d}. {row['team']:15s} {row['attack_strength']:.3f}")
    
    print("\n🛡️ DEFENSE STRENGTHS (Highest = Best Defense)")
    print("=" * 50) 
    for i, (_, row) in enumerate(results_defense.head(20).iterrows()):
        print(f"{i+1:2d}. {row['team']:15s} {row['defense_strength']:.3f}")
    
    return results

# ================================
# STEP 7: PREDICT NEW MATCHES
# ================================

def predict_match_from_strengths(model, team_df, scaler, team_to_idx, home_team, away_team):
    """Predict match using learned team strengths"""
    
    if model is None or home_team not in team_to_idx or away_team not in team_to_idx:
        print(f"❌ Team not found: {home_team} or {away_team}")
        return
    
    model.eval()
    
    # Get team indices
    home_idx = team_to_idx[home_team]
    away_idx = team_to_idx[away_team]
    
    # Get features
    feature_cols = ['avg_goals_for', 'avg_goals_against', 'avg_xg_for', 'avg_xg_against']
    team_features = scaler.transform(team_df[feature_cols].values)
    
    home_features = torch.FloatTensor(team_features[home_idx:home_idx+1])
    away_features = torch.FloatTensor(team_features[away_idx:away_idx+1])
    
    # Predict
    with torch.no_grad():
        home_lambda, away_lambda = model(home_features, away_features)
    
    home_expected = home_lambda.item()
    away_expected = away_lambda.item()
    
    print(f"\n⚽ {home_team} vs {away_team}")
    print(f"🎯 Expected goals: {home_team} {home_expected:.2f} - {away_expected:.2f} {away_team}")
    
    # Win probabilities
    home_win = sum(poisson.pmf(h, home_expected) * sum(poisson.pmf(a, away_expected) 
                  for a in range(h)) for h in range(10))
    draw = sum(poisson.pmf(g, home_expected) * poisson.pmf(g, away_expected) for g in range(10))
    away_win = 1 - home_win - draw
    
    print(f"📊 Win probabilities:")
    print(f"   🏠 {home_team}: {home_win:.1%}")
    print(f"   🤝 Draw: {draw:.1%}")
    print(f"   🛫 {away_team}: {away_win:.1%}")

# ================================
# STEP 8: MAIN EXECUTION
# ================================

if __name__ == "__main__":
    print("🚀 Premier League Team Strength Analysis")
    print("=" * 50)
    
    # Train the model
    model, team_df, scaler, team_to_idx = train_team_strength_model()
    
    if model is not None:
        # Show learned team strengths
        strengths_df = show_team_strengths(model, team_df, scaler)
        
        # Example predictions
        print("\n🔮 EXAMPLE PREDICTIONS")
        print("=" * 50)
        
        teams = list(team_to_idx.keys())
        if len(teams) >= 4:
            predict_match_from_strengths(model, team_df, scaler, team_to_idx, 
                                       teams[0], teams[1])
            predict_match_from_strengths(model, team_df, scaler, team_to_idx, 
                                       teams[2], teams[3])
        
        print(f"\n🎉 Success! Derived team strengths from match history")
        print("\nWhat happened:")
        print("1. ✅ Loaded historical match data and calculated team metrics")
        print("2. ✅ Trained neural network to learn attack/defense strengths")
        print("3. ✅ Network optimized strengths for Poisson goal prediction")
        print("4. ✅ Extracted interpretable team strength rankings")
        print("5. ✅ Can now predict any match using learned strengths")
    
    else:
        print("❌ Training failed - check your data")