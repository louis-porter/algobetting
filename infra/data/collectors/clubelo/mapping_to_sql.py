import pandas as pd
import sqlite3
import numpy as np

conn = sqlite3.connect(r'C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db')

df = pd.read_csv("infra/data/collectors/clubelo/clubelo_fbref_team_name_mapping.csv")

df.to_sql('clubelo_fbref_team_name_mapping', conn, if_exists='replace', index=False)

conn.close()