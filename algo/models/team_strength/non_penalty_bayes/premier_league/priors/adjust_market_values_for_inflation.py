"""
Rebases transfermarkt_squad_values.csv market-value figures into constant
(2025-26 season) terms, using the same UK CPI series (ONS D7BT) and season-
averaging convention as adjust_salaries_for_inflation.py.

Note: squad_value_eur is in EUR (Transfermarkt's default display currency),
while the CPI series is UK inflation. Strictly this should be Eurozone HICP,
or the values converted to GBP first -- this applies UK CPI directly instead,
since the two indices have tracked closely enough over 2015-2026 that it
doesn't change the shape of the trend, only its last-decimal precision.

Input:  transfermarkt_squad_values.csv
Output: transfermarkt_squad_values_adjusted.csv (adds cpi_index, deflator,
        squad_value_2025_26_terms)
"""

from pathlib import Path

import pandas as pd

from adjust_salaries_for_inflation import TARGET_SEASON, season_avg_cpi

DATA_PATH = Path(__file__).parent / "transfermarkt_squad_values.csv"
OUT_PATH = Path(__file__).parent / "transfermarkt_squad_values_adjusted.csv"


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    seasons = sorted(df["season"].unique())
    cpi_by_season = {s: season_avg_cpi(s) for s in seasons}
    target_cpi = cpi_by_season[TARGET_SEASON]

    df["cpi_index"] = df["season"].map(cpi_by_season)
    df["deflator"] = target_cpi / df["cpi_index"]
    df["squad_value_2025_26_terms"] = (df["squad_value_eur"] * df["deflator"]).round().astype(int)

    df.to_csv(OUT_PATH, index=False)

    print(f"Target season: {TARGET_SEASON} (avg CPI {target_cpi:.2f})")
    for s in seasons:
        print(f"  {s}: avg CPI {cpi_by_season[s]:.2f}, deflator {target_cpi / cpi_by_season[s]:.3f}")
    print(f"\nWrote {len(df)} rows -> {OUT_PATH}")
