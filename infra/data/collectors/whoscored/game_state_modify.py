import pandas as pd
import sqlite3

db_path = r'/Users/admin/dev/algobetting/infra/data/db/fotmob.db'
conn = sqlite3.connect(db_path)

# Pull goals only
goals_df = pd.read_sql_query("""
    SELECT 
        matchId,
        startDate,
        homeTeam,
        awayTeam,
        CASE WHEN h_a = 'h' THEN homeTeam ELSE awayTeam END AS team,
        minute,
        second,
        playerName,
        type,
        isGoal,
        goalOwn
    FROM match_events
    WHERE
        division = 'Premier_League'
        AND season = '2025-2026'
        AND playerName != 'None'
        AND isGoal = 1
""", conn)

# Own goals credit the opponent
goals_df['scoring_team'] = goals_df.apply(
    lambda row: row['awayTeam'] if (row['goalOwn'] == 1 and row['team'] == row['homeTeam'])
           else row['homeTeam'] if (row['goalOwn'] == 1 and row['team'] == row['awayTeam'])
           else row['team'],
    axis=1
)

# One row per goal, sorted
goals_df = goals_df.sort_values(['matchId', 'minute', 'second']).reset_index(drop=True)

# Build cumulative home/away score at each goal event
rows = []
for match_id, group in goals_df.groupby('matchId'):
    home = group['homeTeam'].iloc[0]
    away = group['awayTeam'].iloc[0]
    home_score = 0
    away_score = 0

    for _, row in group.iterrows():
        # Score BEFORE this goal (pre-event state)
        pre_home, pre_away = home_score, away_score

        # Update score
        if row['scoring_team'] == home:
            home_score += 1
        else:
            away_score += 1

        rows.append({
            'matchId':        match_id,
            'minute':         row['minute'],
            'second':         row['second'],
            'homeTeam':       home,
            'awayTeam':       away,
            'scoring_team':   row['scoring_team'],
            'playerName':     row['playerName'],
            'goalOwn':        row['goalOwn'],
            # Score after this goal
            'home_score':     home_score,
            'away_score':     away_score,
        })

score_df = pd.DataFrame(rows)

# --- Expand to full game state table (one row per goal-state window) ---
# For each match, generate the state valid FROM each goal until the next
state_rows = []
for match_id, group in score_df.groupby('matchId'):
    home = group['homeTeam'].iloc[0]
    away = group['awayTeam'].iloc[0]

    # State before kick off: 0-0 from minute 0
    checkpoints = [{'minute': 0, 'second': 0, 'home_score': 0, 'away_score': 0}]
    for _, row in group.iterrows():
        checkpoints.append({
            'minute':     row['minute'],
            'second':     row['second'],
            'home_score': row['home_score'],
            'away_score': row['away_score'],
        })

    for i, cp in enumerate(checkpoints):
        next_minute = checkpoints[i + 1]['minute'] if i + 1 < len(checkpoints) else 999
        next_second = checkpoints[i + 1]['second'] if i + 1 < len(checkpoints) else 0
        home_diff = cp['home_score'] - cp['away_score']
        away_diff = cp['away_score'] - cp['home_score']

        def state_label(diff):
            if diff == 0:   return 'Draw'
            elif diff > 0:  return f'Winning {diff}' if diff <= 3 else 'Winning 3+'
            else:           return f'Losing {abs(diff)}' if abs(diff) <= 3 else 'Losing 3+'

        for team, diff in [(home, home_diff), (away, away_diff)]:
            state_rows.append({
                'matchId':        match_id,
                'team':           team,
                'from_minute':    cp['minute'],
                'from_second':    cp['second'],
                'to_minute':      next_minute,
                'to_second':      next_second,
                'home_score':     cp['home_score'],
                'away_score':     cp['away_score'],
                'score':          f"{cp['home_score']}-{cp['away_score']}",
                'game_state':     state_label(diff),
                'goal_diff':      diff,
            })

game_state_df = pd.DataFrame(state_rows)

print(game_state_df.head(20))

# Write to DB
game_state_df.to_sql('match_game_state', conn, if_exists='replace', index=False)
print(f"Written {len(game_state_df)} rows to match_game_state")
conn.close()