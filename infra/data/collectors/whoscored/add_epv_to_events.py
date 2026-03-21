import sqlite3
import pandas as pd
import numpy as np
import json

def add_epv_to_events_table(
    db_name: str = r"/Users/admin/dev/algobetting/infra/data/db/fotmob.db",
    epv_grid_path: str = r"/Users/admin/dev/algobetting/infra/data/collectors/whoscored/EPV_grid.csv"
):
    """
    Reads the match_events table, calculates EPV for successful passes,
    and writes it back as a new 'EPV' column — no re-scraping needed.
    """

    EPV = np.loadtxt(epv_grid_path, delimiter=',')

    def get_epv_at_location(position, epv_grid, attack_direction=1, field_dimen=(106., 68.)):
        x, y = position
        if abs(x) > field_dimen[0] / 2. or abs(y) > field_dimen[1] / 2.:
            return 0.0
        grid = np.fliplr(epv_grid) if attack_direction == -1 else epv_grid
        ny, nx = grid.shape
        dx = field_dimen[0] / float(nx)
        dy = field_dimen[1] / float(ny)
        ix = int((x + field_dimen[0] / 2. - 0.0001) / dx)
        iy = int((y + field_dimen[1] / 2. - 0.0001) / dy)
        return grid[iy, ix]

    def to_metric(val, axis_range=106.):
        """Convert whoscored 0–100 coordinate to metric centred at 0."""
        return (val / 100.0 * axis_range) - (axis_range / 2.0)

    conn = sqlite3.connect(db_name)
    try:
        # Add EPV column if it doesn't exist yet
        cursor = conn.cursor()
        existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(match_events)")}
        if 'EPV' not in existing_cols:
            cursor.execute("ALTER TABLE match_events ADD COLUMN EPV REAL")
            conn.commit()
            print("➕ Added 'EPV' column to match_events")
        else:
            print("ℹ️  'EPV' column already exists — will overwrite NULL values only")

        df = pd.read_sql(
            "SELECT rowid, type, outcomeType, x, y, endX, endY FROM match_events WHERE EPV IS NULL",
            conn
        )
        print(f"📥 Loaded {len(df)} rows with NULL EPV")

        if df.empty:
            print("✅ Nothing to update — all rows already have EPV values")
            return

        # Calculate EPV only for successful passes
        def calc_epv(row):
            if row['type'] == 'Pass' and row['outcomeType'] == 'Successful':
                try:
                    x_m  = to_metric(row['x'],    106.)
                    y_m  = to_metric(row['y'],     68.)
                    ex_m = to_metric(row['endX'],  106.)
                    ey_m = to_metric(row['endY'],  68.)
                    start_epv = get_epv_at_location((x_m, y_m),   EPV)
                    end_epv   = get_epv_at_location((ex_m, ey_m), EPV)
                    return end_epv - start_epv
                except Exception:
                    return None
            return None

        df['EPV'] = df.apply(calc_epv, axis=1)

        # Write back only rows where EPV was actually computed
        updates = df[df['EPV'].notna()][['rowid', 'EPV']]
        print(f"🔄 Updating {len(updates)} rows with EPV values...")

        cursor.executemany(
            "UPDATE match_events SET EPV = ? WHERE rowid = ?",
            [(row['EPV'], row['rowid']) for _, row in updates.iterrows()]
        )
        conn.commit()
        print(f"✅ Successfully wrote {len(updates)} EPV values to match_events")

    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(traceback.format_exc())
    finally:
        conn.close()


if __name__ == "__main__":
    add_epv_to_events_table()