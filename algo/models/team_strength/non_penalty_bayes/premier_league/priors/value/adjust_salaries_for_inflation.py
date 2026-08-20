"""
Rebases prem_salaries.csv wage-bill figures into constant (2025-26 season)
pounds, so a 2015-16 wage bill is directly comparable to a 2025-26 one.

Uses UK CPI (ONS series D7BT, "CPI INDEX 00: ALL ITEMS", 2015=100), monthly
data pulled 2026-08-15 from
https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/mm23
Each PL season's deflator is the average CPI index over that season's months
(Aug of year1 through May of year2), matched to how the season labels in
prem_salaries.csv are constructed. Note this is *general* consumer-price
inflation, not football-specific wage inflation (which has run far ahead of
CPI on TV-deal growth) -- this column answers "what would this wage bill be
worth in today's money", not "was this team relatively big-spending for its
season" (for that, use gross_unadjusted / season mean instead).

Input:  prem_salaries_raw table in infra/data/db/priors.db (manually compiled,
        no scraper -- there's no automated source for wage-bill data)
Output: prem_salaries_adjusted.csv (adds cpi_index, deflator, gross_2025_26_terms)
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[7] / "infra" / "data" / "db" / "priors.db"
OUT_PATH = Path(__file__).parent / "prem_salaries_adjusted.csv"

# ONS series D7BT, CPI INDEX 00: ALL ITEMS (2015=100), monthly.
CPI_MONTHLY = {
    "2015-08": 100.3, "2015-09": 100.2, "2015-10": 100.3, "2015-11": 100.3, "2015-12": 100.3,
    "2016-01": 99.5,  "2016-02": 99.8,  "2016-03": 100.2, "2016-04": 100.2, "2016-05": 100.4,
    "2016-08": 100.9, "2016-09": 101.1, "2016-10": 101.2, "2016-11": 101.4, "2016-12": 101.9,
    "2017-01": 101.4, "2017-02": 102.1, "2017-03": 102.5, "2017-04": 102.9, "2017-05": 103.3,
    "2017-08": 103.8, "2017-09": 104.1, "2017-10": 104.2, "2017-11": 104.6, "2017-12": 104.9,
    "2018-01": 104.4, "2018-02": 104.9, "2018-03": 105.0, "2018-04": 105.4, "2018-05": 105.8,
    "2018-08": 106.5, "2018-09": 106.6, "2018-10": 106.7, "2018-11": 107.0, "2018-12": 107.1,
    "2019-01": 106.3, "2019-02": 106.8, "2019-03": 107.0, "2019-04": 107.6, "2019-05": 107.9,
    "2019-08": 108.4, "2019-09": 108.5, "2019-10": 108.3, "2019-11": 108.5, "2019-12": 108.5,
    "2020-01": 108.2, "2020-02": 108.6, "2020-03": 108.6, "2020-04": 108.5, "2020-05": 108.5,
    "2020-08": 108.6, "2020-09": 109.1, "2020-10": 109.1, "2020-11": 108.9, "2020-12": 109.2,
    "2021-01": 109.0, "2021-02": 109.1, "2021-03": 109.4, "2021-04": 110.1, "2021-05": 110.8,
    "2021-08": 112.1, "2021-09": 112.4, "2021-10": 113.6, "2021-11": 114.5, "2021-12": 115.1,
    "2022-01": 114.9, "2022-02": 115.8, "2022-03": 117.1, "2022-04": 120.0, "2022-05": 120.8,
    "2022-08": 123.1, "2022-09": 123.8, "2022-10": 126.2, "2022-11": 126.7, "2022-12": 127.2,
    "2023-01": 126.4, "2023-02": 127.9, "2023-03": 128.9, "2023-04": 130.4, "2023-05": 131.3,
    "2023-08": 131.3, "2023-09": 132.0, "2023-10": 132.0, "2023-11": 131.7, "2023-12": 132.2,
    "2024-01": 131.5, "2024-02": 132.3, "2024-03": 133.0, "2024-04": 133.5, "2024-05": 133.9,
    "2024-08": 134.3, "2024-09": 134.2, "2024-10": 135.0, "2024-11": 135.1, "2024-12": 135.6,
    "2025-01": 135.4, "2025-02": 136.0, "2025-03": 136.5, "2025-04": 138.2, "2025-05": 138.4,
    "2025-08": 139.3, "2025-09": 139.3, "2025-10": 139.8, "2025-11": 139.5, "2025-12": 140.1,
    "2026-01": 139.5, "2026-02": 140.1, "2026-03": 141.0, "2026-04": 142.1, "2026-05": 142.4,
    # 2026-08 onward not yet released as of the 2026-08-15 pull date -- held flat at the last
    # known reading (2026-05) so season_avg_cpi() doesn't KeyError on the 2026-2027 season.
    # This is a placeholder, not a forecast; re-pull real CPI once ONS publishes it.
    "2026-08": 142.4, "2026-09": 142.4, "2026-10": 142.4, "2026-11": 142.4, "2026-12": 142.4,
    "2027-01": 142.4, "2027-02": 142.4, "2027-03": 142.4, "2027-04": 142.4, "2027-05": 142.4,
}

TARGET_SEASON = "2025-2026"


def season_avg_cpi(season: str) -> float:
    """Average CPI over a season's Aug(year1)-May(year2) months."""
    start_year = int(season.split("-")[0])
    months = (
        [f"{start_year}-{m:02d}" for m in (8, 9, 10, 11, 12)]
        + [f"{start_year + 1}-{m:02d}" for m in (1, 2, 3, 4, 5)]
    )
    values = [CPI_MONTHLY[m] for m in months]
    return sum(values) / len(values)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prem_salaries_raw", conn)
    conn.close()

    seasons = sorted(df["season"].unique())
    cpi_by_season = {s: season_avg_cpi(s) for s in seasons}
    target_cpi = cpi_by_season[TARGET_SEASON]

    df["cpi_index"] = df["season"].map(cpi_by_season)
    df["deflator"] = target_cpi / df["cpi_index"]
    df["gross_2025_26_terms"] = (df["gross_unadjusted"] * df["deflator"]).round().astype(int)

    df.to_csv(OUT_PATH, index=False)

    print(f"Target season: {TARGET_SEASON} (avg CPI {target_cpi:.2f})")
    for s in seasons:
        print(f"  {s}: avg CPI {cpi_by_season[s]:.2f}, deflator {target_cpi / cpi_by_season[s]:.3f}")
    print(f"\nWrote {len(df)} rows -> {OUT_PATH}")
