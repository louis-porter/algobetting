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

class RecentMatchDataScraper:
    def __init__(self, season, days_back=7, headless=True, db_path="team_model_db"):
        self.season = season
        self.days_back = days_back
        self.base_url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
        self.match_data = []
        self.db_path = db_path
        self.setup_driver(headless)
        self.cutoff_date = datetime.now() - timedelta(days=days_back)
        print(f"Scraping matches played after: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
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

    def get_match_data(self, url):
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
                print(f"Teams: {home_team} vs {away_team}")
            else:
                print("Team stats section not found")
                home_team, away_team = None, None
                
            # Extract division
            division_link = soup.find('a', href=lambda x: x and '/comps/' in x and '-Stats' in x)
            division = division_link.text.strip() if division_link else None
            
            # Initialize tables to None
            home_team_table = None
            away_team_table = None
            
            # Extract summary stats
            pattern = re.compile(r'stats_[a-z0-9]+_summary')
            stats_tables = soup.find_all('table', id=pattern)
            print(f"Found {len(stats_tables)} stats tables")
            
            for table in stats_tables:
                caption = table.find("caption")
                if caption and caption.text:
                    caption_text = caption.text.strip()
                    print(f"Table caption: '{caption_text}'")

                    # Assigning teams to each table
                    if home_team and home_team in caption_text:
                        home_team_table = table
                        print(f"✓ Assigned table to home team: {home_team}")
                    elif away_team and away_team in caption_text:
                        away_team_table = table
                        print(f"✓ Assigned table to away team: {away_team}")
                    else:
                        print(f"⚠ Couldn't match table to either team ({home_team} or {away_team})")

            # Verify table assignment
            if home_team_table:
                print(f"Home team ({home_team}) table ID: {home_team_table.get('id')}")
            else:
                print(f"⚠ No table found for home team: {home_team}")

            if away_team_table:
                print(f"Away team ({away_team}) table ID: {away_team_table.get('id')}")
            else:
                print(f"⚠ No table found for away team: {away_team}")

            if not stats_tables:
                print(f"No stats tables found for {url}")
                return None

            # Here you would process the data and return a dataframe
            # For now, just return a dummy value to show success
            return {"match_date": match_date, "home_team": home_team, "away_team": away_team}
                
        except Exception as e:
            print(f"Error processing match data: {e}")
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

class RecentMatchDataScraper:
    def __init__(self, season, days_back=7, headless=True, db_path="team_model_db"):
        self.season = season
        self.days_back = days_back
        self.base_url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
        self.match_data = []
        self.db_path = db_path
        self.setup_driver(headless)
        self.cutoff_date = datetime.now() - timedelta(days=days_back)
        print(f"Scraping matches played after: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
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

    def extract_team_stats_from_table(self, soup, home_team, away_team, match_date, division, url, table_pattern, stat_fields):
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
                'division': division,
                'match_url': url
            }
            
            # Add specific stats from the table footer
            for field in stat_fields:
                cell = home_team_table.find('tfoot').find('td', {'data-stat': field})
                if cell:
                    home_stats[field] = cell.get_text(strip=True)
            
            home_df = pd.DataFrame([home_stats])
        else:
            print(f"⚠ No {table_pattern} table found for home team: {home_team}")
        
        # Extract away team stats
        if away_team_table:
            away_stats = {
                'team': away_team,
                'is_home': False,
                'match_date': match_date,
                'division': division,
                'match_url': url
            }
            
            # Add specific stats from the table footer
            for field in stat_fields:
                cell = away_team_table.find('tfoot').find('td', {'data-stat': field})
                if cell:
                    away_stats[field] = cell.get_text(strip=True)
            
            away_df = pd.DataFrame([away_stats])
        else:
            print(f"⚠ No {table_pattern} table found for away team: {away_team}")
        
        return home_df, away_df

    def get_match_data(self, url):
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
            
            # Extract summary stats
            summary_fields = ['shots', 'shots_on_target', 'xg', 'npxg', 'cards_red']
            home_summary_df, away_summary_df = self.extract_team_stats_from_table(
                soup, home_team, away_team, match_date, division, url, 
                r'stats_[a-z0-9]+_summary', summary_fields
            )
            
            # Extract possession stats
            possession_fields = ['touches_att_pen_area', 'touches_att_3rd']
            home_possession_df, away_possession_df = self.extract_team_stats_from_table(
                soup, home_team, away_team, match_date, division, url, 
                r'stats_[a-z0-9]+_possession', possession_fields
            )

            # Extract penalty stats
            misc_fields = ['pens_won']
            home_misc_df, away_misc_df = self.extract_team_stats_from_table(
                soup, home_team, away_team, match_date, division, url, 
                r'stats_[a-z0-9]+_misc', misc_fields
            )

            # Extract goalkeeper stats with reversed team attribution
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

            # Extract away team's PSxG from home keeper table (reverse attribution)
            if home_keeper_table:
                # Look for the PSxG cell using the correct data-stat attribute
                psxg_cell = home_keeper_table.find('td', {'data-stat': 'gk_psxg'})
                if psxg_cell:
                    away_psxg = psxg_cell.get_text(strip=True)
                    # Add to away_summary_df if needed
                    if not away_summary_df.empty:
                        away_summary_df['psxg'] = away_psxg

            # Extract home team's PSxG from away keeper table (reverse attribution)
            if away_keeper_table:
                # Look for the PSxG cell using the correct data-stat attribute
                psxg_cell = away_keeper_table.find('td', {'data-stat': 'gk_psxg'})
                if psxg_cell:
                    home_psxg = psxg_cell.get_text(strip=True)
                    # Add to home_summary_df if needed
                    if not home_summary_df.empty:
                        home_summary_df['psxg'] = home_psxg
            # Merge all DataFrames into one match DataFrame
            try:
                # First, merge home stats from different tables
                home_df = home_summary_df.copy()
                
                if not home_possession_df.empty:
                    # Make sure we're merging on all common columns to avoid duplicates
                    merge_cols = [col for col in home_df.columns if col in home_possession_df.columns]
                    if merge_cols:
                        home_df = pd.merge(home_df, home_possession_df, on=merge_cols, how='outer')
                    else:
                        # If no common columns, we can just add the new columns
                        for col in home_possession_df.columns:
                            if col not in home_df.columns:
                                home_df[col] = home_possession_df[col].values[0]
                
                if not home_misc_df.empty:
                    # Make sure we're merging on all common columns to avoid duplicates
                    merge_cols = [col for col in home_df.columns if col in home_misc_df.columns]
                    if merge_cols:
                        home_df = pd.merge(home_df, home_misc_df, on=merge_cols, how='outer')
                    else:
                        # If no common columns, we can just add the new columns
                        for col in home_misc_df.columns:
                            if col not in home_df.columns:
                                home_df[col] = home_misc_df[col].values[0]
                
                # Then, merge away stats from different tables
                away_df = away_summary_df.copy()
                
                if not away_possession_df.empty:
                    merge_cols = [col for col in away_df.columns if col in away_possession_df.columns]
                    if merge_cols:
                        away_df = pd.merge(away_df, away_possession_df, on=merge_cols, how='outer')
                    else:
                        for col in away_possession_df.columns:
                            if col not in away_df.columns:
                                away_df[col] = away_possession_df[col].values[0]
                
                if not away_misc_df.empty:
                    merge_cols = [col for col in away_df.columns if col in away_misc_df.columns]
                    if merge_cols:
                        away_df = pd.merge(away_df, away_misc_df, on=merge_cols, how='outer')
                    else:
                        for col in away_misc_df.columns:
                            if col not in away_df.columns:
                                away_df[col] = away_misc_df[col].values[0]
                
                # Combine home and away into a single match DataFrame
                match_df = pd.concat([home_df, away_df], ignore_index=True)
                return match_df

            except Exception as e:
                print(f"Error merging match data: {e}")
                import traceback
                traceback.print_exc()
                return None
                            
        except Exception as e:
            print(f"Error processing match data: {e}")
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

    def scrape_matches(self):
        try:
            print(f"\nStarting scrape for season {self.season}, matches from last {self.days_back} days")
            self.driver.get(self.base_url)
            self.random_delay()
            
            # First, collect all match URLs that meet our criteria
            match_urls = []
            
            table = self.find_fixtures_table()
            if not table:
                raise Exception("Could not find fixtures table")
            
            # Use JavaScript to get all rows and information at once to avoid stale element issues
            script = """
                var results = [];
                var rows = document.querySelectorAll('table tr:not(:first-child)');
                
                for (var i = 0; i < rows.length; i++) {
                    var row = rows[i];
                    var dateCell = row.querySelector('td[data-stat="date"]');
                    var matchReportLink = row.querySelector('td a:contains("Match Report")');
                    
                    if (dateCell && dateCell.textContent.trim()) {
                        var obj = {
                            date: dateCell.textContent.trim(),
                            url: matchReportLink ? matchReportLink.href : null
                        };
                        results.push(obj);
                    }
                }
                
                return results;
            """
            
            # A modified version that will work better with Selenium
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
            
            print(f"Found {len(match_urls)} matches within date range")
            
            # Process each match URL
            matches_processed = 0
            matches_collected = 0
            
            for match_date_text, match_url in match_urls:
                matches_processed += 1
                print(f"\nProcessing match {matches_processed}/{len(match_urls)} from {match_date_text}: {match_url}")
                
                match_data = self.get_match_data(match_url)
                if match_data is not None:
                    self.match_data.append(match_data)
                    matches_collected += 1
                    print(f"Successfully processed match data")
                else:
                    print(f"No data retrieved for match")
                
                # Add delay between matches
                self.random_delay(2, 4)
            
            print(f"\nProcessed {matches_processed} matches within date range, collected data for {matches_collected} matches")
            return True
            
        except Exception as e:
            print(f"An error occurred during scraping: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_results(self):
        if self.match_data:
            # Combine all match data into a single DataFrame
            combined_df = pd.concat(self.match_data, ignore_index=True)
            
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'recent_matches_{self.days_back}days_{timestamp}.csv'
            
            os.makedirs('data', exist_ok=True)
            filepath = os.path.join('data', filename)
            combined_df.to_csv(filepath, index=False)
            print(f"\nResults saved to {filepath}")
            
            # Save to SQLite database
            try:
                conn = sqlite3.connect(self.db_path)
                print(f"\nConnected to database: {self.db_path}")
                
                # Append data to prem_data table
                combined_df.to_sql('prem_data', conn, if_exists='append', index=False)
                print(f"Successfully appended {len(combined_df)} rows to prem_data table")
                
                # Close connection
                conn.close()
            except Exception as e:
                print(f"Error saving to database: {str(e)}")
                
            print(f"\nTotal events collected: {len(combined_df)}")
            return True
        else:
            print("\nNo match data collected")
            return False

    def cleanup(self):
        if hasattr(self, 'driver'):
            self.driver.quit()
            
    def run(self):
        try:
            if self.scrape_matches():
                self.save_results()
        finally:
            self.cleanup()
            print("\nScript completed")

if __name__ == "__main__":
    # Set the season and number of days to look back
    season = "2024-2025"  # Update with current season
    days_back = 7  # Get matches from last 14 days
    db_path = "team_model_db"  # SQLite database file path
    
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
    scraper = RecentMatchDataScraper(season, days_back=days_back, headless=True, db_path=db_path)
    scraper.run()
    def save_results(self):
        if self.match_data:
            # Combine all match data into a single DataFrame
            combined_df = pd.concat(self.match_data, ignore_index=True)
            
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'recent_matches_{self.days_back}days_{timestamp}.csv'
            
            os.makedirs('data', exist_ok=True)
            filepath = os.path.join('data', filename)
            combined_df.to_csv(filepath, index=False)
            print(f"\nResults saved to {filepath}")
            
            # Save to SQLite database
            try:
                conn = sqlite3.connect(self.db_path)
                print(f"\nConnected to database: {self.db_path}")
                
                # Append data to prem_data table
                combined_df.to_sql('prem_data', conn, if_exists='append', index=False)
                print(f"Successfully appended {len(combined_df)} rows to prem_data table")
                
                # Close connection
                conn.close()
            except Exception as e:
                print(f"Error saving to database: {str(e)}")
                
            print(f"\nTotal events collected: {len(combined_df)}")
            return True
        else:
            print("\nNo match data collected")
            return False

    def cleanup(self):
        if hasattr(self, 'driver'):
            self.driver.quit()
            
    def run(self):
        try:
            if self.scrape_matches():
                self.save_results()
        finally:
            self.cleanup()
            print("\nScript completed")

if __name__ == "__main__":
    # Set the season and number of days to look back
    season = "2024-2025"  # Update with current season
    days_back = 7  # Get matches from last 14 days
    db_path = "team_model_db"  # SQLite database file path
    
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
    scraper = RecentMatchDataScraper(season, days_back=days_back, headless=True, db_path=db_path)
    scraper.run()