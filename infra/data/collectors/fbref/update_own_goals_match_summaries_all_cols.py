import pandas as pd
import sqlite3

conn = sqlite3.connect(r'C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db')

df = pd.read_sql_query("""
                        SELECT *
                        FROM 
                            fbref_match_summary_v2
                       """, conn)

# Rename columns for clarity if needed (optional)
df.columns = [col.strip().lower() for col in df.columns]  # if you want lowercase

# Define column names
match_url_col = "match_url"
summary_goals_col = "summary_goals"
opp_summary_goals_col = "opp_summary_goals"

# Make sure goals columns are numeric
df[summary_goals_col] = pd.to_numeric(df[summary_goals_col], errors='coerce')
df[opp_summary_goals_col] = pd.to_numeric(df[opp_summary_goals_col], errors='coerce')

# Function to fix opp_summary_goals
def fix_opp_summary_goals(group):
    # Should be two rows per match
    if len(group) != 2:
        return group  # skip malformed matches

    # Get the other team's summary_goals
    goals = group[summary_goals_col].values
    # Update opp_summary_goals
    group[opp_summary_goals_col] = goals[::-1]  # Flip the array
    return group

# Apply fix to each match group
df_fixed = df.groupby(match_url_col, group_keys=False).apply(fix_opp_summary_goals)

print(df_fixed[df_fixed["match_url"] == 'https://fbref.com/en/matches/7289bcdf/Fulham-Crystal-Palace-February-22-2025-Premier-League'][["summary_goals",  "opp_summary_goals"]].head())

df_fixed.to_sql("fbref_match_summary_v2_fixed", conn, if_exists="replace", index=False)
