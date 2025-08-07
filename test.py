import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FootballDataset(Dataset):
    def __init__(self, matches_df):
        """
        Expected DataFrame columns:
        - home_team_id, away_team_id (integers 0 to n_teams-1)
        - is_home (1 for home advantage, 0 otherwise)
        - goals_home, goals_away (target variables)
        - Optional: xG_home, xG_away, shots_home, shots_away, etc. (for features)
        """
        self.home_teams = torch.tensor(matches_df['home_team_id'].values, dtype=torch.long)
        self.away_teams = torch.tensor(matches_df['away_team_id'].values, dtype=torch.long)
        self.is_home = torch.tensor(matches_df['is_home'].values, dtype=torch.float32)
        self.goals_home = torch.tensor(matches_df['goals_home'].values, dtype=torch.float32)
        self.goals_away = torch.tensor(matches_df['goals_away'].values, dtype=torch.float32)
        
        # Optional: include observable features as additional context
        feature_cols = ['xG_home', 'xG_away', 'shots_home', 'shots_away']
        if all(col in matches_df.columns for col in feature_cols):
            self.features = torch.tensor(matches_df[feature_cols].values, dtype=torch.float32)
        else:
            self.features = None
    
    def __len__(self):
        return len(self.home_teams)
    
    def __getitem__(self, idx):
        item = {
            'home_team': self.home_teams[idx],
            'away_team': self.away_teams[idx], 
            'is_home': self.is_home[idx],
            'goals_home': self.goals_home[idx],
            'goals_away': self.goals_away[idx]
        }
        if self.features is not None:
            item['features'] = self.features[idx]
        return item

class TalentBottleneckModel(nn.Module):
    def __init__(self, n_teams, feature_dim=0, hidden_dim=128):
        super().__init__()
        self.n_teams = n_teams
        
        # Input: team indices + home advantage + optional features
        input_dim = 1 + feature_dim  # home advantage + features
        
        # Hidden layers to process context
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Talent parameter layers (the bottleneck!)
        # Each team gets attack and defense parameters
        self.attack_embedding = nn.Embedding(n_teams, 1)
        self.defense_embedding = nn.Embedding(n_teams, 1)
        
        # Global parameters
        self.home_advantage = nn.Parameter(torch.tensor(0.0))
        
        # Optional: context-dependent adjustments to talent
        if feature_dim > 0:
            self.talent_adjustment = nn.Linear(hidden_dim, 2)  # adjust attack/defense
        else:
            self.talent_adjustment = None
            
        # Initialize embeddings
        nn.init.normal_(self.attack_embedding.weight, 0, 0.1)
        nn.init.normal_(self.defense_embedding.weight, 0, 0.1)
    
    def forward(self, home_team, away_team, is_home, features=None):
        batch_size = home_team.size(0)
        
        # Base talent parameters
        home_attack = self.attack_embedding(home_team).squeeze(-1)  # [batch_size]
        home_defense = self.defense_embedding(home_team).squeeze(-1)
        away_attack = self.attack_embedding(away_team).squeeze(-1)  
        away_defense = self.defense_embedding(away_team).squeeze(-1)
        
        # Process context features
        context_input = is_home.unsqueeze(-1)  # [batch_size, 1]
        if features is not None:
            context_input = torch.cat([context_input, features], dim=-1)
            
        hidden_output = self.hidden(context_input)
        
        # Optional: adjust talents based on context
        if self.talent_adjustment is not None:
            adjustments = self.talent_adjustment(hidden_output)  # [batch_size, 2]
            home_attack_adj = adjustments[:, 0]
            home_defense_adj = adjustments[:, 1]
        else:
            home_attack_adj = 0
            home_defense_adj = 0
        
        # Final talent parameters (the interpretable bottleneck)
        final_home_attack = home_attack + home_attack_adj
        final_home_defense = home_defense + home_defense_adj
        
        # Calculate Poisson rates using classic attack - defense formula
        lambda_home = torch.exp(final_home_attack - away_defense + self.home_advantage * is_home)
        lambda_away = torch.exp(away_attack - final_home_defense)
        
        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'talents': {
                'home_attack': final_home_attack,
                'home_defense': final_home_defense, 
                'away_attack': away_attack,
                'away_defense': away_defense,
                'home_advantage': self.home_advantage
            }
        }

def poisson_loss(pred_lambda, actual_goals):
    """Negative log-likelihood for Poisson distribution"""
    # Clamp lambda to avoid numerical issues
    pred_lambda = torch.clamp(pred_lambda, min=1e-8, max=10.0)
    return -torch.sum(actual_goals * torch.log(pred_lambda) - pred_lambda - torch.lgamma(actual_goals + 1))

def train_model(model, train_loader, val_loader, epochs=1000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch['home_team'], batch['away_team'], batch['is_home'], 
                          batch.get('features'))
            
            # Poisson loss for both home and away goals
            loss_home = poisson_loss(output['lambda_home'], batch['goals_home'])
            loss_away = poisson_loss(output['lambda_away'], batch['goals_away'])
            loss = loss_home + loss_away
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['home_team'], batch['away_team'], batch['is_home'],
                              batch.get('features'))
                loss_home = poisson_loss(output['lambda_home'], batch['goals_home'])
                loss_away = poisson_loss(output['lambda_away'], batch['goals_away'])
                val_loss += (loss_home + loss_away).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

def extract_team_talents(model, n_teams):
    """Extract the learned attack/defense parameters for analysis"""
    model.eval()
    with torch.no_grad():
        team_ids = torch.arange(n_teams)
        attacks = model.attack_embedding(team_ids).squeeze().numpy()
        defenses = model.defense_embedding(team_ids).squeeze().numpy()
        home_adv = model.home_advantage.item()
        
    return {
        'attacks': attacks,
        'defenses': defenses, 
        'home_advantage': home_adv
    }

# Example usage
if __name__ == "__main__":
    import sqlite3
    
    # Load real Premier League data
    def load_real_data():
        """Load match data from your database"""
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
        
        # Create team ID mapping
        all_teams = sorted(list(set(df['team'].unique()) | set(df['opp_team'].unique())))
        team_to_id = {team: idx for idx, team in enumerate(all_teams)}
        id_to_team = {idx: team for team, idx in team_to_id.items()}
        
        # Convert to required format
        processed_data = pd.DataFrame({
            'home_team_id': df['team'].map(team_to_id),
            'away_team_id': df['opp_team'].map(team_to_id),
            'is_home': 1,  # All records are home matches from the query
            'goals_home': df['goals_for'],
            'goals_away': df['goals_against'],
            # Optional features
            'xG_home': df['xg_for'],
            'xG_away': df['xg_against'],
            'shots_home': df['shots_for'],
            'shots_away': df['shots_against'],
            'date': df['date']
        })
        
        # Remove any invalid matches
        processed_data = processed_data.dropna()
        processed_data = processed_data[processed_data['home_team_id'] != processed_data['away_team_id']]
        
        return processed_data, team_to_id, id_to_team
    
    # Load the real data
    real_data, team_mapping, id_mapping = load_real_data()
    n_teams = len(team_mapping)
    
    print(f"Loaded {len(real_data)} Premier League matches")
    print(f"Number of teams: {n_teams}")
    print(f"Teams: {list(team_mapping.keys())}")
    print(f"Sample data:")
    print(real_data.head())
    
    
    # Split data chronologically (use early matches for training, later for validation)
    real_data = real_data.sort_values('date')
    train_size = int(0.8 * len(real_data))
    train_data = real_data.iloc[:train_size]
    val_data = real_data.iloc[train_size:]
    
    print(f"\nTraining on {len(train_data)} matches, validating on {len(val_data)} matches")
    
    # Create datasets (remove possession since it's not in your data)
    train_dataset = FootballDataset(train_data)
    val_dataset = FootballDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    feature_dim = 4  # xG_home, xG_away, shots_home, shots_away
    model = TalentBottleneckModel(n_teams=n_teams, feature_dim=feature_dim, hidden_dim=64)
    
    print("\nTraining model on Premier League data...")
    train_model(model, train_loader, val_loader, epochs=1000, lr=0.001)
    
    # Extract learned talents and show team rankings
    talents = extract_team_talents(model, n_teams)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'team': [id_mapping[i] for i in range(n_teams)],
        'attack_talent': talents['attacks'],
        'defense_talent': talents['defenses']
    })
    
    # Sort by attack talent
    attack_ranking = results_df.sort_values('attack_talent', ascending=False)
    defense_ranking = results_df.sort_values('defense_talent', ascending=True)  # Lower is better defense
    
    print(f"\nLearned Model Parameters:")
    print(f"Home advantage: {talents['home_advantage']:.3f}")
    
    print(f"\nTop 5 Attack Teams:")
    print(attack_ranking.head(20)[['team', 'attack_talent']])

    
    print(f"\nTop 5 Defense Teams (lowest defense values):")
    print(defense_ranking.head(20)[['team', 'defense_talent']])
    
    
    # Show some sample predictions vs actual
    model.eval()
    sample_matches = val_data.head(10)
    print(f"\nSample Predictions vs Actual (validation set):")
    print("Home Team vs Away Team | Pred Goals | Actual Goals")
    print("-" * 55)
    
    with torch.no_grad():
        for _, match in sample_matches.iterrows():
            home_team_name = id_mapping[match['home_team_id']]
            away_team_name = id_mapping[match['away_team_id']]
            
            # Get prediction
            output = model(
                torch.tensor([match['home_team_id']], dtype=torch.long),
                torch.tensor([match['away_team_id']], dtype=torch.long),
                torch.tensor([match['is_home']], dtype=torch.float32),
                torch.tensor([[match['xG_home'], match['xG_away'], 
                             match['shots_home'], match['shots_away']]], dtype=torch.float32)
            )
            
            pred_home = output['lambda_home'].item()
            pred_away = output['lambda_away'].item()
            actual_home = match['goals_home']
            actual_away = match['goals_away']
            
            print(f"{home_team_name:<15} vs {away_team_name:<15} | {pred_home:.1f}-{pred_away:.1f} | {actual_home:.0f}-{actual_away:.0f}")