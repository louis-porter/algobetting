import fotmob.fotmob_season_downloader
from fotmob.fotmob_etl_database_non_penalty import main
from whoscored.whoscored_scraper import process_epv_data
from whoscored.main import main
from datetime import datetime


def collect_all_strength_data(season, year, start_year, start_month, start_day, end_year, end_month, end_day):
    #fotmob.fotmob_season_downloader.main()
    main(season=season, year=year)
    process_epv_data(
        start_date=datetime(start_year, start_month, start_day),
        end_date=datetime(end_year, end_month, end_day),
        competition='england-premier-league',
        season='2025/2026',
        season_label='2025-2026',
        division='Premier_League'
    )




collect_all_strength_data('Premier_League', '2025-2026', 
                          2026, 3, 25, 
                          2026, 3, 26)

