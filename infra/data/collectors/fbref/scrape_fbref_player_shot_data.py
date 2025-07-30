import pandas as pd
import time
import random
import os
import sqlite3
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests
from bs4 import BeautifulSoup
import html
import re
import hashlib
# No argparse needed

# Try to import lxml, use html.parser as fallback
try:
    import lxml
    DEFAULT_PARSER = 'lxml'
except ImportError:
    print("lxml parser not available, using html.parser instead")
    DEFAULT_PARSER = 'html.parser'

class RecentMatchDataScraper:
    def __init__(self, season=None, seasons=None, league=None, days_back=7, headless=True, db_path="team_model_db", table_name="prem_data", specific_url=None):
        # Handle multiple seasons input
        if seasons is not None:
            self.seasons = seasons if isinstance(seasons, list) else [seasons]
            self.season = None  # Will be set during processing
        elif season is not None:
            self.seasons = [season]
            self.season = season
        else:
            self.seasons = []
            self.season = season
            
        self.league = league
        self.days_back = days_back
        self.specific_url = specific_url
        
        if specific_url:
            print(f"Specific URL mode: Will scrape only {specific_url}")
        else:
            if not self.seasons or not league:
                raise ValueError("Season(s) and league must be provided when not using specific URL")
            if len(self.seasons) > 1:
                print(f"Multi-season mode: Will scrape {len(self.seasons)} seasons: {', '.join(self.seasons)}")
            else:
                print(f"Single season mode: Will scrape season {self.seasons[0]}")
                
        self.match_data = []
        self.db_path = db_path
        self.table_name = table_name
        self.setup_driver(headless)
        
        if not specific_url:
            self.cutoff_date = datetime.now() - timedelta(days=days_back)
            print(f"Scraping matches played after: {self.cutoff_date.strftime('%Y-%m-%d')}")
    
    def create_record_hash(self, player, team, match_date, match_url):
        """Create a unique hash for a player's match record to detect duplicates"""
        # Combine key fields that make a record unique
        unique_string = f"{player}|{team}|{match_date}|{match_url}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def get_existing_hashes(self):
        """Get all existing record hashes from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f'''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self.table_name}'
            ''')
            
            if not cursor.fetchone():
                conn.close()
                return set()  # No table exists yet, no duplicates possible
            
            # Check if record_hash column exists
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'record_hash' not in columns:
                # Add record_hash column if it doesn't exist
                cursor.execute(f'ALTER TABLE {self.table_name} ADD COLUMN record_hash TEXT')
                conn.commit()
                print("Added record_hash column to existing table")
                conn.close()
                return set()  # No hashes exist yet
            
            # Get all existing hashes
            cursor.execute(f'SELECT record_hash FROM {self.table_name} WHERE record_hash IS NOT NULL')
            existing_hashes = {row[0] for row in cursor.fetchall()}
            
            conn.close()
            print(f"Found {len(existing_hashes)} existing records in database")
            return existing_hashes
            
        except Exception as e:
            print(f"Error getting existing hashes: {e}")
            return set()
    
    def filter_duplicates(self, df):
        """Filter out duplicate records from DataFrame"""
        if df.empty:
            return df
            
        # Get existing hashes from database
        existing_hashes = self.get_existing_hashes()
        
        # Add record_hash column to DataFrame
        df['record_hash'] = df.apply(
            lambda row: self.create_record_hash(
                row['player'], 
                row['team'], 
                row['match_date'], 
                row['match_url']
            ), 
            axis=1
        )
        
        # Filter out duplicates
        original_count = len(df)
        df_filtered = df[~df['record_hash'].isin(existing_hashes)].copy()
        duplicates_removed = original_count - len(df_filtered)
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate records")
        
        return df_filtered

    def setup_driver(self, headless):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.maximize_window()

    def random_delay(self, min_seconds=3, max_seconds=5):
        time.sleep(random.uniform(min_seconds, max_seconds))

    def extract_player_stats_from_table(self, soup, home_team, away_team, match_date, division, url, table_pattern, stat_fields=None):
        # Find tables matching the pattern
        pattern = re.compile(table_pattern)
        stats_tables = soup.find_all('table', id=pattern)
        
        home_team_table = None
        away_team_table = None
        
        # Assign tables to respective teams
        for table in stats_tables:
            caption = table.find("caption")
            if caption and caption.text:
                caption_text = caption.text.strip()
                
                if home_team and home_team in caption_text:
                    home_team_table = table
                elif away_team and away_team in caption_text:
                    away_team_table = table
        
        home_players_df = pd.DataFrame()
        away_players_df = pd.DataFrame()
        
        # Extract home team player stats
        if home_team_table:
            # Get headers to know which stats are available
            headers = []
            header_row = home_team_table.find('thead').find_all('th')
            for header in header_row:
                data_stat = header.get('data-stat')
                if data_stat:
                    headers.append(data_stat)
            
            # If stat_fields is None, use all available stats except basic info fields
            if stat_fields is None:
                # Exclude common non-stat fields
                excluded_fields = []#['player', 'shirtnumber', 'nationality', 'position', 'age', 'minutes', 'games', 'games_starts']
                stat_fields = [h for h in headers if h not in excluded_fields]
            
            # Get player rows - each tbody row is a player
            player_rows = home_team_table.find('tbody').find_all('tr')
            home_players_data = []
            
            for row in player_rows:
                # Skip summary rows or rows without data-stat
                if 'class' in row.attrs and 'divider' in row['class']:
                    continue
                    
                # Extract player name and other data
                player_data = {
                    'team': home_team,
                    'opponent': away_team,
                    'is_home': True,
                    'match_date': match_date,
                    'division': division, 
                    'season': self.season if self.season else self._extract_season_from_url(url),
                    'match_url': url
                }
                
                # Get player name
                player_cell = row.find('th', {'data-stat': 'player'})
                if player_cell:
                    player_data['player'] = player_cell.get_text(strip=True)
                
                # Extract all available stats for the player
                for field in headers:
                    cell = row.find('td', {'data-stat': field})
                    if cell:
                        player_data[field] = cell.get_text(strip=True)
                
                home_players_data.append(player_data)
            
            if home_players_data:
                home_players_df = pd.DataFrame(home_players_data)
        else:
            print(f"⚠ No {table_pattern} table found for home team: {home_team}")
        
        # Extract away team player stats
        if away_team_table:
            # Get headers to know which stats are available
            headers = []
            header_row = away_team_table.find('thead').find_all('th')
            for header in header_row:
                data_stat = header.get('data-stat')
                if data_stat:
                    headers.append(data_stat)
            
            # If stat_fields is None, use all available stats except basic info fields
            if stat_fields is None:
                # Exclude common non-stat fields
                excluded_fields = []#['player', 'shirtnumber', 'nationality', 'position', 'age', 'minutes', 'games', 'games_starts']
                stat_fields = [h for h in headers if h not in excluded_fields]
            
            # Get player rows - each tbody row is a player
            player_rows = away_team_table.find('tbody').find_all('tr')
            away_players_data = []
            
            for row in player_rows:
                # Skip summary rows or rows without data-stat
                if 'class' in row.attrs and 'divider' in row['class']:
                    continue
                    
                # Extract player name and other data
                player_data = {
                    'team': away_team,
                    'opponent': home_team,
                    'is_home': False,
                    'match_date': match_date,
                    'division': division,
                    'season': self.season if self.season else self._extract_season_from_url(url),
                    'match_url': url
                }
                
                # Get player name
                player_cell = row.find('th', {'data-stat': 'player'})
                if player_cell:
                    player_data['player'] = player_cell.get_text(strip=True)
                
                # Extract all available stats for the player
                for field in headers:
                    cell = row.find('td', {'data-stat': field})
                    if cell:
                        player_data[field] = cell.get_text(strip=True)
                
                away_players_data.append(player_data)
            
            if away_players_data:
                away_players_df = pd.DataFrame(away_players_data)
        else:
            print(f"⚠ No {table_pattern} table found for away team: {away_team}")
        
        return home_players_df, away_players_df

    def _extract_season_from_url(self, url):
        """Extract season from match URL if season wasn't provided during initialization"""
        match = re.search(r'/(\d{4}-\d{4})/', url)
        if match:
            return match.group(1)
        return "Unknown"

    def get_match_player_data(self, url):
        self.random_delay()
        
        try:
            print(f"Fetching player data with Selenium from: {url}")
            # Use Selenium to navigate to the page
            self.driver.get(url)
            self.random_delay(2, 3)  # Allow page to fully load
            
            # Get the page source and parse it with BeautifulSoup
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, DEFAULT_PARSER)

            # Extract match date
            venue_time = soup.find('span', class_='venuetime')
            match_date = venue_time['data-venue-date'] if venue_time else None
            
            # Skip if match is older than cutoff date (only if we're not using a specific URL)
            if not self.specific_url and match_date:
                match_datetime = datetime.strptime(match_date, '%Y-%m-%d')
                if match_datetime < self.cutoff_date:
                    print(f"Skipping match from {match_date} (older than {self.days_back} days)")
                    return None
            
            # Extract teams
            team_stats = soup.find('div', id='team_stats_extra')
            if team_stats:
                teams = team_stats.find_all('div', class_='th')
                teams = [t.text.strip() for t in teams if t.text.strip() != '']
                teams = list(dict.fromkeys(teams))  # Remove duplicates while preserving order
                home_team = teams[0] if len(teams) > 0 else None
                away_team = teams[1] if len(teams) > 1 else None
            else:
                # Alternative method to extract teams
                title_element = soup.find('h1')
                if title_element:
                    title_text = title_element.text.strip()
                    teams_match = re.match(r'(.+?)\s+vs\.\s+(.+?)(?:\s+Match|$)', title_text)
                    if teams_match:
                        home_team = teams_match.group(1).strip()
                        away_team = teams_match.group(2).strip()
                    else:
                        home_team, away_team = None, None
                else:
                    home_team, away_team = None, None
                
            # Extract division
            division_link = soup.find('a', href=lambda x: x and '/comps/' in x and '-Stats' in x)
            division = division_link.text.strip() if division_link else None
            
            # Create dictionaries to store player data for each team
            # Key will be (player, team, match_date)
            home_player_data = {}
            away_player_data = {}
            
            # Helper function to update player dictionaries
            def update_player_dict(player_dict, player_df):
                if player_df.empty:
                    return
                    
                for _, row in player_df.iterrows():
                    player_key = (row['player'], row['team'], row['match_date'])
                    
                    if player_key not in player_dict:
                        # Initialize with all metadata
                        player_dict[player_key] = {
                            'player': row['player'],
                            'team': row['team'],
                            'opponent': row['opponent'],
                            'is_home': row['is_home'],
                            'match_date': row['match_date'],
                            'division': row['division'],
                            'season': row.get('season', self.season if self.season else self._extract_season_from_url(url)),
                            'match_url': row['match_url']
                        }
                    
                    # Add all stats from this table
                    for col, value in row.items():
                        if col not in ['player', 'team', 'opponent', 'is_home', 'match_date', 'division', 'season', 'match_url']:
                            # Only update if the field doesn't exist or was empty/NaN
                            if col not in player_dict[player_key] or pd.isna(player_dict[player_key][col]):
                                player_dict[player_key][col] = value
            
            # Extract and process table types
            table_patterns = [
                ('summary', r'stats_[a-z0-9]+_summary'),
                ('possession', r'stats_[a-z0-9]+_possession'),
                ('passing', r'stats_[a-z0-9]+_passing'),
                ('passing_types', r'stats_[a-z0-9]+_passing_types'),
                ('defense', r'stats_[a-z0-9]+_defense'),
                ('misc', r'stats_[a-z0-9]+_misc')
                #('keeper', r'keeper_stats_[a-z0-9]+')
            ]
            
            for stat_type, pattern in table_patterns:
                home_df, away_df = self.extract_player_stats_from_table(
                    soup, home_team, away_team, match_date, division, url, pattern
                )
                
                # Update player dictionaries
                update_player_dict(home_player_data, home_df)
                update_player_dict(away_player_data, away_df)
            
            # Convert dictionaries to DataFrames
            home_players_df = pd.DataFrame(list(home_player_data.values())) if home_player_data else pd.DataFrame()
            away_players_df = pd.DataFrame(list(away_player_data.values())) if away_player_data else pd.DataFrame()
            
            # Combine home and away
            all_players_df = pd.concat([home_players_df, away_players_df], ignore_index=True) if not (home_players_df.empty and away_players_df.empty) else pd.DataFrame()
            
            if not all_players_df.empty:
                print(f"Collected data for {len(all_players_df)} players")
                return all_players_df
            else:
                print("No player data found")
                return None
                
        except Exception as e:
            print(f"Error processing player match data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def find_fixtures_table(self):
        try:
            table_selectors = [
                f"table#sched_{self.season}_9_1",
                "table.stats_table.sortable",
                "//table[contains(@class, 'stats_table')]"
            ]
            
            for selector in table_selectors:
                try:
                    if selector.startswith("//"):
                        table = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, selector))
                        )
                    else:
                        table = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                    return table
                except TimeoutException:
                    continue
                    
            raise Exception("Could not find fixtures table")
            
        except Exception as e:
            print(f"Error finding fixtures table: {e}")
            return None

    def scrape_specific_match(self):
        """Scrape data from a specific match URL"""
        try:
            print(f"\nStarting scrape for specific match: {self.specific_url}")
            
            match_data = self.get_match_player_data(self.specific_url)
            if match_data is not None:
                self.match_data.append(match_data)
                print(f"Successfully processed match data")
                return True
            else:
                print(f"No data retrieved for match")
                return False
                
        except Exception as e:
            print(f"An error occurred during specific match scraping: {e}")
            import traceback
            traceback.print_exc()
            return False

    def scrape_season_matches(self, season):
        """Scrape matches for a specific season"""
        self.season = season
        base_url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-{self.league}-Scores-and-Fixtures"
        
        print(f"\nScraping season {season} from: {base_url}")
        
        try:
            self.driver.get(base_url)
            self.random_delay()
            
            # First, collect all match URLs that meet our criteria
            match_urls = []
            
            table = self.find_fixtures_table()
            if not table:
                raise Exception("Could not find fixtures table")
            
            # Use JavaScript to get all rows and information at once to avoid stale element issues
            script = """
                var results = [];
                var rows = document.querySelectorAll('table tr');
                
                for (var i = 1; i < rows.length; i++) {  // Skip header row
                    var row = rows[i];
                    var dateCell = row.querySelector('td[data-stat="date"]');
                    var matchReportLink = row.querySelector('td a');
                    
                    if (dateCell && dateCell.textContent.trim()) {
                        var hasMatchReport = false;
                        var reportUrl = null;
                        
                        if (matchReportLink) {
                            if (matchReportLink.textContent.trim() === 'Match Report') {
                                hasMatchReport = true;
                                reportUrl = matchReportLink.href;
                            } else {
                                // Check other links in the row
                                var allLinks = row.querySelectorAll('td a');
                                for (var j = 0; j < allLinks.length; j++) {
                                    if (allLinks[j].textContent.trim() === 'Match Report') {
                                        hasMatchReport = true;
                                        reportUrl = allLinks[j].href;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        results.push({
                            date: dateCell.textContent.trim(),
                            url: reportUrl,
                            hasReport: hasMatchReport
                        });
                    }
                }
                
                return results;
            """
            
            match_data = self.driver.execute_script(script)
            
            # Filter matches within our date range
            match_urls = []
            for match in match_data:
                if not match['date'] or not match['url'] or not match['hasReport']:
                    continue
                    
                try:
                    match_date = datetime.strptime(match['date'], '%Y-%m-%d')
                except ValueError:
                    try:
                        match_date = datetime.strptime(match['date'], '%a %d/%m/%Y')
                    except ValueError:
                        print(f"Could not parse date: {match['date']}")
                        continue
                
                if match_date >= self.cutoff_date:
                    match_urls.append((match['date'], match['url']))
            
            # Reverse the list to get most recent matches first
            match_urls.reverse()
            
            print(f"Found {len(match_urls)} matches within date range for season {season}")
            
            # Process each match URL
            matches_processed = 0
            matches_collected = 0
            
            for match_date_text, match_url in match_urls:
                matches_processed += 1
                print(f"\nProcessing match {matches_processed}/{len(match_urls)} from {match_date_text}: {match_url}")
                
                match_data = self.get_match_player_data(match_url)
                if match_data is not None:
                    self.match_data.append(match_data)
                    matches_collected += 1
                    print(f"Successfully processed match data")
                else:
                    print(f"No data retrieved for match")
                
                # Add delay between matches
                self.random_delay(2, 4)
            
            print(f"\nSeason {season}: Processed {matches_processed} matches within date range, collected data for {matches_collected} matches")
            return True
            
        except Exception as e:
            print(f"An error occurred during scraping season {season}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def scrape_matches(self):
        """Scrape multiple matches - can handle multiple seasons"""
        if self.specific_url:
            return self.scrape_specific_match()
        
        total_seasons_processed = 0
        
        # Process each season
        for season in self.seasons:
            print(f"\n{'='*60}")
            print(f"PROCESSING SEASON: {season}")
            print(f"{'='*60}")
            
            success = self.scrape_season_matches(season)
            if success:
                total_seasons_processed += 1
            
            # Add delay between seasons
            if len(self.seasons) > 1:
                print(f"Pausing before next season...")
                self.random_delay(5, 8)
        
        print(f"\n{'='*60}")
        print(f"COMPLETED: Processed {total_seasons_processed}/{len(self.seasons)} seasons")
        print(f"{'='*60}")
        
        return total_seasons_processed > 0

    def save_results(self):
        """Save results to both CSV and SQLite database with duplicate detection"""
        if not self.match_data:
            print("\nNo match data collected")
            return False
            
        try:
            # Combine all match data into a single DataFrame
            combined_df = pd.concat(self.match_data, ignore_index=True)
            print(f"\nTotal records before duplicate filtering: {len(combined_df)}")
            
            # Filter out duplicates
            combined_df = self.filter_duplicates(combined_df)
            
            if combined_df.empty:
                print("No new records to save after duplicate filtering")
                return True
            
            print(f"Records to save after duplicate filtering: {len(combined_df)}")
            
            # Save to SQLite database
            try:
                # Create the database file if it doesn't exist
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if table exists, create it if not
                cursor.execute(f'''
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='{self.table_name}'
                ''')
                
                if not cursor.fetchone():
                    print(f"Creating {self.table_name} table as it doesn't exist")
                    # Generate table creation SQL based on DataFrame columns
                    # This is a simplistic approach - you might want to define specific types
                    columns = []
                    for col in combined_df.columns:
                        if col in ['match_date']:
                            columns.append(f'"{col}" TEXT')
                        elif 'is_' in col:
                            columns.append(f'"{col}" BOOLEAN')
                        elif col == 'record_hash':
                            columns.append(f'"{col}" TEXT UNIQUE')  # Make hash unique
                        elif any(num in col for num in ['shots', 'xg', 'touches', 'psxg']):
                            columns.append(f'"{col}" REAL')
                        else:
                            columns.append(f'"{col}" TEXT')
                    
                    create_table_sql = f'''
                        CREATE TABLE {self.table_name} (
                            {', '.join(columns)}
                        )
                    '''
                    cursor.execute(create_table_sql)
                    conn.commit()
                else:
                    # Check if record_hash column exists in existing table
                    cursor.execute(f"PRAGMA table_info({self.table_name})")
                    columns = [column[1] for column in cursor.fetchall()]
                    
                    if 'record_hash' not in columns:
                        cursor.execute(f'ALTER TABLE {self.table_name} ADD COLUMN record_hash TEXT')
                        conn.commit()
                        print("Added record_hash column to existing table")
                
                # Append data to the specified table
                combined_df.to_sql(self.table_name, conn, if_exists='append', index=False)
                print(f"Successfully appended {len(combined_df)} new rows to {self.table_name} table")
                
                # Close connection
                conn.close()
                
                print(f"\nTotal new events collected: {len(combined_df)}")
                return True
            
            except Exception as e:
                print(f"Error saving to database: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()
            
    def run(self):
        """Run the full scraping process"""
        try:
            if self.scrape_matches():
                self.save_results()
        finally:
            self.cleanup()
            print("\nScript completed")

if __name__ == "__main__":
    # Set variables here - modify these as needed
    
    # SINGLE SEASON MODE
    # season = "2024-2025"  # Update with current season
    
    # MULTIPLE SEASONS MODE - Use this instead of single season
    seasons = ["2024-2025", "2023-2024", "2022-2023", "2021-2022"]  # List of seasons to scrape
    
    league = "Premier-League"
    days_back = 5000  # Get matches from last 365 days
    table_name = "fbref_player_stats"  # Table name in the database
    db_path = r"C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db"  # SQLite database file path
    headless = True  # Run in headless mode
    
    # If you want to scrape a specific match, set specific_url
    # Leave as None to use schedule mode (scrape recent matches)
    specific_url = None
    # Example: specific_url = "https://fbref.com/en/matches/12345/TeamA-TeamB"
    
    # Check and notify about required packages
    required_packages = {
        'lxml': 'Optional but recommended: pip install lxml',
        'selenium': 'Required for web scraping: pip install selenium',
        'requests': 'Required for HTTP requests: pip install requests',
        'beautifulsoup4': 'Required for HTML parsing: pip install beautifulsoup4',
        'pandas': 'Required for data handling: pip install pandas',
        'sqlite3': 'Built-in to Python for database operations'
    }
    
    for package, install_msg in required_packages.items():
        try:
            if package != 'sqlite3':  # sqlite3 is built-in, no need to import check
                __import__(package)
                print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed. {install_msg}")
    
    print("\nStarting scraper...")
    
    if specific_url:
        # Use specific URL mode
        print(f"Specific URL mode: Will scrape only {specific_url}")
        scraper = RecentMatchDataScraper(
            days_back=days_back,
            db_path=db_path, 
            table_name=table_name, 
            headless=headless,
            specific_url=specific_url
        )
    else:
        # Use schedule mode with multiple seasons
        print(f"Schedule mode: Will scrape matches from the past {days_back} days")
        scraper = RecentMatchDataScraper(
            seasons=seasons,  # Pass list of seasons
            league=league,
            days_back=days_back,
            db_path=db_path, 
            table_name=table_name, 
            headless=headless
        )
        
    scraper.run()