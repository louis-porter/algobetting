import pandas as pd
import sqlite3
import numpy as np

conn = sqlite3.connect(r'C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db')

try:
    # Drop table if it exists
    conn.execute("DROP TABLE IF EXISTS clubelo_features")
    
    # Create new table
    sql = """
    CREATE TABLE clubelo_features AS 
    SELECT DISTINCT
        c.Club as clubelo_team,
        f.fbref_name as fbref_team,
        c.Elo as elo,
        c.[From] as start_date,
        LEAD([From]) OVER (PARTITION BY Club ORDER BY [From]) AS end_date
    FROM
        clubelo_ratings c
    JOIN
        clubelo_fbref_team_name_mapping f
            ON f.clubelo_name = c.Club
    """
    
    conn.execute(sql)
    conn.commit()
    
finally:
    conn.close()