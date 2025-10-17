import fotmob.fotmob_season_downloader
import fotmob.fotmob_etl_database
from whoscored.whoscored_scraper import process_epv_data
from datetime import datetime


def collect_all_strength_data(season, year):
    fotmob.fotmob_season_downloader.main()
    fotmob.fotmob_etl_database.main(season=season, year=year)
    process_epv_data(
        start_date=datetime(2024, 8, 1),
        end_date=datetime(2024, 12, 31),
        competition='england-premier-league',
        season='2024/2025',
        season_label='2024-2025',
        division='Premier_League'
    )

