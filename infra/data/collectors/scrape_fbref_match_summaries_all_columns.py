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

# Try to import lxml, use html.parser as fallback
try:
    import lxml
    DEFAULT_PARSER = 'lxml'
except ImportError:
    print("lxml parser not available, using html.parser instead")
    DEFAULT_PARSER = 'html.parser'

class MultiSeasonMatchDataScraper:
    def __init__(self, seasons, league, league_id, days_back=7, headless=True, db_path="team_model_db", table_name="prem_data"):
        self.seasons = seasons if isinstance(seasons, list) else [seasons]
        self.league = league
        self.league_id = league_id
        self.days_back = days_back
        self.match_data = []
        self.season_match_data = []  # Store data for current season only
        self.db_path = db_path
        self.table_name = table_name
        self.setup_driver(headless)
        self.cutoff_date = datetime.now() - timedelta(days=days_back)
        self.existing_urls = self.get_existing_urls()
        self.processed_urls_this_session = set()  # Track URLs processed in current session
        print(f"Scraping matches played after: {self.cutoff_date.strftime('%Y-%m-%d')}")
        print(f"Found {len(self.existing_urls)} existing match URLs in the database")
        print(f"Seasons to process: {', '.join(self.seasons)}")
        
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

    def get_existing_urls(self):
        """Get list of match URLs that already exist in the database"""
        try:
            # Check if database and table exist
            if not os.path.exists(self.db_path):
                print(f"Database {self.db_path} does not exist yet. Creating new database.")
                return set()
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f'''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self.table_name}'
            ''')
            
            if not cursor.fetchone():
                print(f"Table {self.table_name} doesn't exist yet. Will be created when saving data.")
                conn.close()
                return set()
            
            # Check if match_url column exists in the table
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'match_url' not in columns:
                print(f"No 'match_url' column in {self.table_name} table. Cannot check for duplicates.")
                conn.close()
                return set()
            
            # Get existing URLs
            cursor.execute(f"SELECT DISTINCT match_url FROM {self.table_name}")
            urls = {row[0] for row in cursor.fetchall() if row[0]}
            
            conn.close()
            return urls
            
        except Exception as e:
            print(f"Error fetching existing URLs: {e}")
            return set()

    def is_duplicate_url(self, url):
        """Check if URL is duplicate (in DB or already processed this session)"""
        return url in self.existing_urls or url in self.processed_urls_this_session

    def add_processed_url(self, url):
        """Add URL to processed set and existing URLs"""
        self.processed_urls_this_session.add(url)
        self.existing_urls.add(url)

    def random_delay(self, min_seconds=3, max_seconds=5):
        time.sleep(random.uniform(min_seconds, max_seconds))

    def extract_all_stats_from_table(self, soup, home_team, away_team, match_date, division, url, table_pattern, table_prefix):
        """Extract ALL statistics from a table type and prefix with table name"""
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
        
        home_df = pd.DataFrame()
        away_df = pd.DataFrame()
        
        # Extract home team stats
        if home_team_table:
            home_stats = {
                'team': home_team,
                'is_home': True,
                'match_date': match_date,
                'division': self.league,
                'season': self.current_season,
                'match_url': url
            }
            
            # Extract ALL stats from the table footer
            tfoot = home_team_table.find('tfoot')
            if tfoot:
                # Get all cells with data-stat attributes
                stat_cells = tfoot.find_all('td', {'data-stat': True})
                for cell in stat_cells:
                    stat_name = cell.get('data-stat')
                    stat_value = cell.get_text(strip=True)
                    
                    # Skip empty values and certain columns we don't want
                    if stat_value and stat_name not in ['date', 'time', 'comp', 'round', 'day_of_week', 'venue', 'result', 'gf', 'ga', 'opponent', 'poss', 'attendance', 'captain', 'formation', 'referee']:
                        # Prefix the column name with table type
                        prefixed_name = f"{table_prefix}_{stat_name}"
                        home_stats[prefixed_name] = stat_value
            
            home_df = pd.DataFrame([home_stats])
        else:
            print(f"⚠ No {table_pattern} table found for home team: {home_team}")
        
        # Extract away team stats
        if away_team_table:
            away_stats = {
                'team': away_team,
                'is_home': False,
                'match_date': match_date,
                'division': self.league,
                'season': self.current_season,
                'match_url': url
            }
            
            # Extract ALL stats from the table footer
            tfoot = away_team_table.find('tfoot')
            if tfoot:
                # Get all cells with data-stat attributes
                stat_cells = tfoot.find_all('td', {'data-stat': True})
                for cell in stat_cells:
                    stat_name = cell.get('data-stat')
                    stat_value = cell.get_text(strip=True)
                    
                    # Skip empty values and certain columns we don't want
                    if stat_value and stat_name not in ['date', 'time', 'comp', 'round', 'day_of_week', 'venue', 'result', 'gf', 'ga', 'opponent', 'poss', 'attendance', 'captain', 'formation', 'referee']:
                        # Prefix the column name with table type
                        prefixed_name = f"{table_prefix}_{stat_name}"
                        away_stats[prefixed_name] = stat_value
            
            away_df = pd.DataFrame([away_stats])
        else:
            print(f"⚠ No {table_pattern} table found for away team: {away_team}")
        
        return home_df, away_df

    def extract_keeper_stats(self, soup, home_team, away_team, match_date, division, url):
        """Extract goalkeeper stats with proper team attribution (PSxG goes to opposing team)"""
        keeper_pattern = re.compile(r'keeper_stats_[a-z0-9]+')
        keeper_tables = soup.find_all('table', id=keeper_pattern)

        home_keeper_table = None
        away_keeper_table = None

        # Find the keeper tables for each team
        for table in keeper_tables:
            caption = table.find("caption")
            if caption and caption.text:
                caption_text = caption.text.strip()
                
                if home_team and home_team in caption_text:
                    home_keeper_table = table
                elif away_team and away_team in caption_text:
                    away_keeper_table = table

        home_psxg = None
        away_psxg = None

        def extract_psxg_from_keeper_table(table):
            """Extract PSxG from a keeper table (checking both tfoot and tbody)"""
            # First try tfoot
            tfoot = table.find('tfoot')
            if tfoot:
                psxg_cell = tfoot.find('td', {'data-stat': 'gk_psxg'})
                if psxg_cell:
                    return psxg_cell.get_text(strip=True)
            
            # If not in tfoot, try tbody
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
                for row in rows:
                    psxg_cell = row.find('td', {'data-stat': 'gk_psxg'})
                    if psxg_cell:
                        return psxg_cell.get_text(strip=True)
            
            return None

        # Extract away team's PSxG from home keeper table (reverse attribution)
        if home_keeper_table:
            away_psxg = extract_psxg_from_keeper_table(home_keeper_table)

        # Extract home team's PSxG from away keeper table (reverse attribution)
        if away_keeper_table:
            home_psxg = extract_psxg_from_keeper_table(away_keeper_table)

        return home_psxg, away_psxg

    def get_match_data(self, url):
        # Check for duplicate before processing
        if self.is_duplicate_url(url):
            print(f"Skipping duplicate URL: {url}")
            return None
            
        self.random_delay()
        
        try:
            print(f"Fetching data with Selenium from: {url}")
            # Use Selenium to navigate to the page
            self.driver.get(url)
            self.random_delay(2, 3)  # Allow page to fully load
            
            # Get the page source and parse it with BeautifulSoup
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, DEFAULT_PARSER)

            # Extract match date
            venue_time = soup.find('span', class_='venuetime')
            match_date = venue_time['data-venue-date'] if venue_time else None
            
            # Skip if match is older than cutoff date
            if match_date:
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
                home_team, away_team = None, None
                
            # Extract division
            division_link = soup.find('a', href=lambda x: x and '/comps/' in x and '-Stats' in x)
            division = division_link.text.strip() if division_link else None
            
            # Define table configurations with patterns and prefixes
            table_configs = [
                (r'stats_[a-z0-9]+_summary', 'summary'),
                (r'stats_[a-z0-9]+_possession', 'possession'),
                (r'stats_[a-z0-9]+_misc', 'misc'),
                (r'stats_[a-z0-9]+_passing_types', 'pass_types'),
                (r'stats_[a-z0-9]+_passing', 'passing'),
                (r'stats_[a-z0-9]+_defense', 'defense')
            ]
            
            # Extract all table types
            all_home_data = {}
            all_away_data = {}
            
            for table_pattern, table_prefix in table_configs:
                home_df, away_df = self.extract_all_stats_from_table(
                    soup, home_team, away_team, match_date, division, url, 
                    table_pattern, table_prefix
                )
                
                # Add data to dictionaries
                if not home_df.empty:
                    all_home_data.update(home_df.iloc[0].to_dict())
                if not away_df.empty:
                    all_away_data.update(away_df.iloc[0].to_dict())

            # Extract goalkeeper PSxG data
            home_psxg, away_psxg = self.extract_keeper_stats(
                soup, home_team, away_team, match_date, division, url
            )
            
            # Add PSxG to the respective teams
            if home_psxg:
                all_home_data['keeper_psxg'] = home_psxg
            if away_psxg:
                all_away_data['keeper_psxg'] = away_psxg

            # Create DataFrames from the collected data
            home_df = pd.DataFrame([all_home_data]) if all_home_data else pd.DataFrame()
            away_df = pd.DataFrame([all_away_data]) if all_away_data else pd.DataFrame()
            
            # Create copies for opposition stats
            home_df_copy = home_df.copy()
            away_df_copy = away_df.copy()
            
            if not home_df.empty and not away_df.empty:
                # Create dictionaries for opposition stats to avoid DataFrame fragmentation
                home_opp_stats = {}
                away_opp_stats = {}
                
                # Add away team name as opp_team to home team
                if 'team' in away_df_copy.columns:
                    home_opp_stats['opp_team'] = away_df_copy['team'].values[0]

                # Add away team stats as opposition stats to home team
                for col in away_df_copy.columns:
                    if col not in ['match_date', 'team', 'opponent', 'division', 'match_url', 'season', 'is_home']:
                        home_opp_stats[f'opp_{col}'] = away_df_copy[col].values[0]

                # Add home team name as opp_team to away team
                if 'team' in home_df_copy.columns:
                    away_opp_stats['opp_team'] = home_df_copy['team'].values[0]
                
                # Add home team stats as opposition stats to away team
                for col in home_df_copy.columns:
                    if col not in ['match_date', 'team', 'opponent', 'division', 'match_url', 'season', 'is_home']:
                        away_opp_stats[f'opp_{col}'] = home_df_copy[col].values[0]
                
                # Convert dictionaries to DataFrames and concatenate efficiently
                home_opp_df = pd.DataFrame([home_opp_stats])
                away_opp_df = pd.DataFrame([away_opp_stats])
                
                # Concatenate horizontally (axis=1) to add opposition columns
                home_df = pd.concat([home_df, home_opp_df], axis=1)
                away_df = pd.concat([away_df, away_opp_df], axis=1)
                
                # HANDLE OWN GOALS: Add opponent's own goals to team's goals
                # Check if we have own goals and regular goals columns
                if 'misc_own_goals' in away_df_copy.columns and 'summary_goals' in home_df.columns:
                    try:
                        away_og = pd.to_numeric(away_df_copy['misc_own_goals'].values[0], errors='coerce') or 0
                        home_goals = pd.to_numeric(home_df['summary_goals'].values[0], errors='coerce') or 0
                        home_df['summary_goals'] = str(home_goals + away_og)
                    except Exception as e:
                        print(f"Error handling own goals for home team: {e}")
                
                if 'misc_own_goals' in home_df_copy.columns and 'summary_goals' in away_df.columns:
                    try:
                        home_og = pd.to_numeric(home_df_copy['misc_own_goals'].values[0], errors='coerce') or 0
                        away_goals = pd.to_numeric(away_df['summary_goals'].values[0], errors='coerce') or 0
                        away_df['summary_goals'] = str(away_goals + home_og)
                    except Exception as e:
                        print(f"Error handling own goals for away team: {e}")

            # Remove own_goals columns to prevent database issues
            columns_to_remove = ['misc_own_goals', 'opp_misc_own_goals']
            for col in columns_to_remove:
                if col in home_df.columns:
                    home_df = home_df.drop(columns=[col])
                if col in away_df.columns:
                    away_df = away_df.drop(columns=[col])

            
            # Combine home and away into a single match DataFrame
            match_df = pd.concat([home_df, away_df], ignore_index=True)

            # Remove duplicate columns that appear under different prefixes
            match_df = self.remove_duplicate_columns(match_df)

            # Convert all stats to numeric types
            match_df = self.convert_stats_to_numeric(match_df)

            # Mark this URL as processed
            self.add_processed_url(url)

            print(f"Extracted {len(match_df.columns)} total columns from match (after removing duplicates)")
            return match_df

        except Exception as e:
            print(f"Error processing match data: {e}")
            import traceback
            traceback.print_exc()
            return None
        

    def convert_stats_to_numeric(self, df):
        """Convert all stat columns to numeric types, keeping text columns as text"""
        
        # Columns that should remain as text
        text_columns = [
            'team', 'division', 'season', 'match_url', 'match_date', 'opp_team'
        ]
        
        # Boolean columns that should remain boolean
        boolean_columns = [
            'is_home'
        ]
        
        for col in df.columns:
            # Skip text and boolean columns
            if col in text_columns or col in boolean_columns:
                continue
                
            # Convert all other columns to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df   

    def remove_duplicate_columns(self, df):
        """Automatically remove duplicate columns that exist under different table prefixes"""
        
        # Track base column names and which columns to keep/drop
        base_columns = {}  # base_name -> first_column_with_that_base
        columns_to_drop = []
        
        # Process all columns to find duplicates
        for col in df.columns:
            # Skip core columns that should never be removed
            if col in ['team', 'is_home', 'match_date', 'division', 'season', 'match_url', 'opp_team']:
                continue
                
            # Extract base name by removing table prefix
            base_name = None
            
            if col.startswith('opp_'):
                # Handle opposition columns: opp_prefix_base -> opp_base
                opp_part = col[4:]  # Remove 'opp_'
                if '_' in opp_part:
                    prefix, base = opp_part.split('_', 1)
                    base_name = f"opp_{base}"
            elif '_' in col:
                # Handle regular columns: prefix_base -> base
                prefix, base = col.split('_', 1)
                base_name = base
            
            # If we found a base name, check for duplicates
            if base_name:
                if base_name in base_columns:
                    # Duplicate found - mark current column for removal
                    columns_to_drop.append(col)
                else:
                    # First occurrence - keep this one
                    base_columns[base_name] = col
        
        # Remove duplicate columns
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
        return df

    def find_fixtures_table(self):
        try:
            # First, ensure we're on the "All Rounds" view by clicking the appropriate tab
            self.ensure_all_rounds_view()
            
            # Wait a moment for the view to load
            self.random_delay(1, 2)
            
            table_selectors = [
                f"table#sched_{self.current_season}_9_1",
                "table.stats_table.sortable",
                "//table[contains(@class, 'stats_table')]",
                "//table[@id and contains(@id, 'sched_')]"
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

    def ensure_all_rounds_view(self):
        """Ensure we're viewing the 'All Rounds' tab rather than specific months/rounds"""
        try:
            # Look for tab navigation elements
            tab_selectors = [
                "a.sr_preset[data-show*='all_sched']",  # All rounds tab
                "a.sr_preset:contains('All Rounds')",   # Alternative selector
                f"a.sr_preset[data-show*='{self.current_season}_{self.league_id}_1']",  # Season-specific all rounds
                "//a[contains(@class, 'sr_preset') and contains(text(), 'All')]",  # XPath for "All" text
                "//a[contains(@class, 'sr_preset') and contains(@data-show, 'all_sched')]"  # XPath for all_sched
            ]
            
            print(f"Ensuring 'All Rounds' view is selected for season {self.current_season}")
            
            for selector in tab_selectors:
                try:
                    if selector.startswith("//"):
                        element = self.driver.find_element(By.XPATH, selector)
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    # Check if element is visible and clickable
                    if element.is_displayed():
                        print(f"Found tab element: {element.get_attribute('outerHTML')[:100]}...")
                        self.driver.execute_script("arguments[0].click();", element)
                        print(f"Clicked 'All Rounds' tab using selector: {selector}")
                        return True
                        
                except Exception as e:
                    print(f"Tab selector {selector} failed: {e}")
                    continue
            
            # Alternative approach: look for any tab that shows all matches
            try:
                # Find all tab elements
                all_tabs = self.driver.find_elements(By.CSS_SELECTOR, "a.sr_preset")
                print(f"Found {len(all_tabs)} tab elements")
                
                for tab in all_tabs:
                    tab_text = tab.text.strip().lower()
                    data_show = tab.get_attribute('data-show') or ''
                    
                    print(f"Tab text: '{tab_text}', data-show: '{data_show}'")
                    
                    # Look for tabs that might show all matches
                    if any(keyword in tab_text for keyword in ['all', 'championship', 'rounds']) or \
                       'all_sched' in data_show.lower():
                        print(f"Clicking tab with text: '{tab_text}'")
                        self.driver.execute_script("arguments[0].click();", tab)
                        return True
                        
            except Exception as e:
                print(f"Alternative tab selection failed: {e}")
            
            # Final fallback: try to click the first tab if no specific "All" tab found
            try:
                first_tab = self.driver.find_element(By.CSS_SELECTOR, "a.sr_preset")
                print(f"Fallback: clicking first available tab")
                self.driver.execute_script("arguments[0].click();", first_tab)
                return True
            except Exception as e:
                print(f"Fallback tab click failed: {e}")
            
            print("Warning: Could not find or click 'All Rounds' tab, proceeding with current view")
            return False
            
        except Exception as e:
            print(f"Error ensuring All Rounds view: {e}")
            return False
        
    def filter_columns_to_match_table(self, combined_df):
        """Filter DataFrame to only include columns that exist in the database table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f'''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self.table_name}'
            ''')
            
            if not cursor.fetchone():
                # Table doesn't exist, return original DataFrame (it will be created)
                conn.close()
                return combined_df
            
            # Get existing table columns
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            existing_columns = {info[1] for info in cursor.fetchall()}
            conn.close()
            
            # Filter DataFrame to only include existing columns
            df_columns = set(combined_df.columns)
            missing_columns = df_columns - existing_columns
            
            if missing_columns:
                print(f"Ignoring {len(missing_columns)} missing columns: {sorted(missing_columns)}")
                # Keep only columns that exist in the table
                columns_to_keep = [col for col in combined_df.columns if col in existing_columns]
                filtered_df = combined_df[columns_to_keep]
                return filtered_df
            
            return combined_df
            
        except Exception as e:
            print(f"Error filtering columns: {e}")
            return combined_df

    def save_season_results(self, season):
        """Save results for a specific season to the database"""
        if not self.season_match_data:
            print(f"\nNo new match data collected for season {season}")
            return False
            
        try:
            # Combine all match data for this season into a single DataFrame
            combined_df = pd.concat(self.season_match_data, ignore_index=True)
            
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
                    # Generate table creation SQL based on DataFrame columns with better typing
                    columns = []
                    for col in combined_df.columns:
                        if col in ['team', 'division', 'season', 'match_url', 'match_date', 'opp_team']:
                            columns.append(f'"{col}" TEXT')
                        elif col in ['is_home']:
                            columns.append(f'"{col}" BOOLEAN')
                        else:
                            # All other columns (stats) should be numeric
                            columns.append(f'"{col}" REAL')
                    
                    create_table_sql = f'''
                        CREATE TABLE {self.table_name} (
                            {', '.join(columns)}
                        )
                    '''
                    cursor.execute(create_table_sql)
                    conn.commit()
                else:
                    # Table exists, filter DataFrame to match existing columns
                    combined_df = self.filter_columns_to_match_table(combined_df)
                
                conn.close()
                
                # Now save the filtered data
                conn = sqlite3.connect(self.db_path)
                combined_df.to_sql(self.table_name, conn, if_exists='append', index=False)
                print(f"Successfully saved {len(combined_df)} rows for season {season} to {self.table_name} table")
                print(f"Total columns saved: {len(combined_df.columns)}")
                
                # Close connection
                conn.close()
                
                # Add to overall match data and clear season data
                self.match_data.extend(self.season_match_data)
                self.season_match_data = []  # Clear for next season
                
                return True
            
            except Exception as e:
                print(f"Error saving season {season} to database: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"Error saving season {season} results: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    

    def scrape_season_matches(self, season):
        """Scrape matches for a specific season"""
        self.current_season = season
        self.season_match_data = []  # Reset season data
        base_url = f"https://fbref.com/en/comps/{self.league_id}/{season}/schedule/{season}-{self.league}-Scores-and-Fixtures"
        
        try:
            print(f"\n{'='*60}")
            print(f"Starting scrape for season {season}")
            print(f"URL: {base_url}")
            print(f"{'='*60}")
            
            self.driver.get(base_url)
            self.random_delay()
            
            # First, collect all match URLs that meet our criteria
            match_urls = []
            
            table = self.find_fixtures_table()
            if not table:
                print(f"Could not find fixtures table for season {season}")
                return 0, 0, 0
            
            # Use JavaScript to get all rows and information at once to avoid stale element issues
            script = """
                var results = [];
                
                // First, ensure we're looking at the right table (visible one)
                var tables = document.querySelectorAll('table[id*="sched"]');
                var activeTable = null;
                
                for (var t = 0; t < tables.length; t++) {
                    var table = tables[t];
                    var tableStyle = window.getComputedStyle(table);
                    
                    // Check if table is visible
                    if (tableStyle.display !== 'none' && tableStyle.visibility !== 'hidden') {
                        activeTable = table;
                        break;
                    }
                }
                
                if (!activeTable) {
                    // Fallback to first table if none found to be specifically visible
                    activeTable = tables[0];
                }
                
                if (activeTable) {
                    console.log('Using table:', activeTable.id);
                    var rows = activeTable.querySelectorAll('tr');
                    
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
                    
                    console.log('Found', results.length, 'potential matches in table');
                } else {
                    console.log('No schedule table found');
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
            matches_skipped = 0
            
            for match_date_text, match_url in match_urls:
                matches_processed += 1
                print(f"\nProcessing match {matches_processed}/{len(match_urls)} from {match_date_text}: {match_url}")
                
                # Check if the match URL already exists in the database or was processed this session
                if self.is_duplicate_url(match_url):
                    print(f"Skipping match - already exists in database or processed this session: {match_url}")
                    matches_skipped += 1
                    continue
                
                match_data = self.get_match_data(match_url)
                if match_data is not None:
                    self.season_match_data.append(match_data)  # Add to season-specific data
                    matches_collected += 1
                    print(f"Successfully processed match data")
                else:
                    print(f"No data retrieved for match")
                
                # Add delay between matches
                self.random_delay(2, 4)
            
            print(f"\nSeason {season} Summary:")
            print(f"Processed {matches_processed} matches within date range")
            print(f"Skipped {matches_skipped} matches (duplicates)")
            print(f"Collected data for {matches_collected} new matches")
            
            # Save season data to database immediately
            if matches_collected > 0:
                print(f"\nSaving season {season} data to database...")
                if self.save_season_results(season):
                    print(f"✓ Season {season} data saved successfully")
                else:
                    print(f"✗ Failed to save season {season} data")
            
            return matches_processed, matches_collected, matches_skipped
            
        except Exception as e:
            print(f"An error occurred during scraping season {season}: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0, 0

    def scrape_all_seasons(self):
        """Scrape matches for all specified seasons"""
        total_processed = 0
        total_collected = 0
        total_skipped = 0
        
        for i, season in enumerate(self.seasons, 1):
            print(f"\n{'*'*80}")
            print(f"PROCESSING SEASON {i}/{len(self.seasons)}: {season}")
            print(f"{'*'*80}")
            
            processed, collected, skipped = self.scrape_season_matches(season)
            total_processed += processed
            total_collected += collected
            total_skipped += skipped
            
            # Add delay between seasons
            if i < len(self.seasons):
                print(f"\nWaiting before next season...")
                self.random_delay(5, 8)
        
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY FOR ALL SEASONS")
        print(f"{'='*80}")
        print(f"Total matches processed: {total_processed}")
        print(f"Total matches skipped (duplicates): {total_skipped}")
        print(f"Total new matches collected: {total_collected}")
        print(f"Seasons processed: {', '.join(self.seasons)}")
        
        return total_collected > 0

    def save_results(self):
        """Legacy method - now data is saved after each season"""
        if not self.match_data:
            print("\nNo match data to save (all data already saved per season)")
            return True
        else:
            print(f"\nAll {len(self.match_data)} matches have been saved to database during processing")
            return True

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()
            
    def run(self):
        """Run the full scraping process for all seasons"""
        try:
            if self.scrape_all_seasons():
                self.save_results()  # This now just confirms data was saved
        finally:
            self.cleanup()
            print("\nScript completed")

if __name__ == "__main__":
    # Set the seasons and other parameters
    seasons = ["2022-2023"] # List of seasons to scrape
    league = "Serie B"
    league_id = 18
    days_back = 90000  # Get matches from last X days
    table_name = "fbref_match_summary_v2"  # Table name in the database
    db_path = r"C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db"  # SQLite database file path
     
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
    
    print("\nStarting multi-season scraper...")
    print(f"Seasons to process: {', '.join(seasons)}")
    
    scraper = MultiSeasonMatchDataScraper(
        seasons=seasons, 
        league=league, 
        league_id=league_id, 
        days_back=days_back, 
        db_path=db_path, 
        table_name=table_name, 
        headless=True
    )
    scraper.run()