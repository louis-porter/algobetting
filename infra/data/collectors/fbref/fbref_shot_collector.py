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
from urllib.parse import urljoin

# Try to import lxml, use html.parser as fallback
try:
    import lxml
    DEFAULT_PARSER = 'lxml'
    print("✓ Using lxml parser")
except ImportError:
    print("⚠ lxml parser not available, using html.parser instead")
    print("  Recommend: pip install lxml")
    DEFAULT_PARSER = 'html.parser'

class ImprovedMatchDataScraper:
    def __init__(self, season, days_back=7, headless=True, db_path="team_model_db"):
        self.season = season
        self.days_back = days_back
        self.base_url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
        self.match_data = []
        self.db_path = db_path
        self.cutoff_date = datetime.now() - timedelta(days=days_back)
        self.setup_session()
        self.setup_driver(headless)
        print(f"Scraping matches played after: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
    def setup_session(self):
        """Setup requests session with proper headers and retry logic"""
        self.session = requests.Session()
        
        # Rotate through different user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
    def setup_driver(self, headless):
        """Setup Selenium driver with anti-detection measures"""
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')  # Use new headless mode
        
        # Anti-detection options
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--no-first-run')
        options.add_argument('--no-default-browser-check')
        options.add_argument('--disable-default-apps')
        
        # Randomize user agent
        user_agent = random.choice(self.user_agents)
        options.add_argument(f'user-agent={user_agent}')
        
        # Exclude automation flags
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=options)
            # Execute script to hide automation indicators
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.set_window_size(1920, 1080)
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("Make sure ChromeDriver is installed and in PATH")
            raise

    def smart_delay(self, min_seconds=5, max_seconds=10):
        """Implement smarter delays with exponential backoff on errors"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def get_with_retry(self, url, max_retries=3):
        """Make HTTP request with retry logic and rotating user agents"""
        for attempt in range(max_retries):
            try:
                # Rotate user agent
                self.session.headers['User-Agent'] = random.choice(self.user_agents)
                
                # Add random delay
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(1, 3)
                    print(f"  Retry {attempt + 1} after {delay:.1f}s delay...")
                    time.sleep(delay)
                
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 403:
                    print(f"  403 Forbidden - attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        # Longer delay for 403 errors
                        time.sleep(random.uniform(10, 20))
                        continue
                    else:
                        print(f"  Max retries reached for 403 error")
                        return None
                elif response.status_code == 429:
                    print(f"  Rate limited - attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(30, 60))
                        continue
                    else:
                        return None
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                print(f"  Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    return None
                    
        return None

    def get_match_data(self, url):
        """Get match data using improved request handling"""
        print(f"  Fetching: {url}")
        
        response = self.get_with_retry(url)
        if not response:
            print(f"  Failed to fetch match data")
            return None
            
        try:
            soup = BeautifulSoup(response.text, DEFAULT_PARSER)

            # Extract match date
            venue_time = soup.find('span', class_='venuetime')
            match_date = venue_time['data-venue-date'] if venue_time else None
            
            # Skip if match is older than cutoff date
            if match_date:
                match_datetime = datetime.strptime(match_date, '%Y-%m-%d')
                if match_datetime < self.cutoff_date:
                    print(f"  Skipping - match from {match_date} is older than {self.days_back} days")
                    return None
            
            # Extract teams from the scorebox
            scorebox = soup.find('div', class_='scorebox')
            if scorebox:
                team_divs = scorebox.find_all('div', recursive=False)
                teams = []
                for div in team_divs:
                    team_link = div.find('a')
                    if team_link and '/squads/' in team_link.get('href', ''):
                        teams.append(team_link.get_text(strip=True))
                
                home_team = teams[0] if len(teams) > 0 else None
                away_team = teams[1] if len(teams) > 1 else None
            else:
                # Fallback method
                team_stats = soup.find('div', id='team_stats_extra')
                if team_stats:
                    teams = team_stats.find_all('div', class_='th')
                    teams = [t.text.strip() for t in teams if t.text.strip() != '']
                    teams = list(dict.fromkeys(teams))
                    home_team = teams[0] if len(teams) > 0 else None
                    away_team = teams[1] if len(teams) > 1 else None
                else:
                    home_team, away_team = None, None
                
            # Extract division
            division_link = soup.find('a', href=lambda x: x and '/comps/' in x and '-Stats' in x)
            division = division_link.text.strip() if division_link else "Premier League"
            
            # Get shots data
            shots_table = soup.find('table', id='shots_all')
            if not shots_table:
                print(f"  No shots table found")
                return None
            
            # Process shots table
            df = self.process_shots_table(shots_table, soup, url, match_date, 
                                        home_team, away_team, division)
            
            if df is not None and len(df) > 0:
                print(f"  Successfully extracted {len(df)} events")
                return df
            else:
                print(f"  No valid data extracted")
                return None
                
        except Exception as e:
            print(f"  Error processing match data: {str(e)}")
            return None

    def process_shots_table(self, shots_table, soup, url, match_date, 
                           home_team, away_team, division):
        """Process the shots table and extract data"""
        try:
            # Get headers from the second row
            header_rows = shots_table.find_all('tr')[:2]
            if len(header_rows) < 2:
                print(f"  Invalid shots table structure")
                return None
            
            headers = []
            for th in header_rows[1].find_all(['th', 'td']):
                header_text = th.get_text(strip=True)
                headers.append(header_text if header_text else f"Column_{len(headers)}")
            
            # Make headers unique
            unique_headers = []
            header_counts = {}
            for header in headers:
                if header in header_counts:
                    header_counts[header] += 1
                    unique_headers.append(f"{header}_{header_counts[header]}")
                else:
                    header_counts[header] = 1
                    unique_headers.append(header)
            
            # Extract shot data rows
            rows_data = []
            tbody = shots_table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    cols = tr.find_all(['th', 'td'])
                    row_data = []
                    for col in cols:
                        value = col.get_text(strip=True)
                        if value == '':
                            value = "0"
                        row_data.append(value)
                    if len(row_data) == len(unique_headers):
                        rows_data.append(row_data)
            
            if not rows_data:
                print(f"  No shot data found in table")
                return None
            
            # Create shots DataFrame
            shots_df = pd.DataFrame(rows_data, columns=unique_headers)
            
            # Find required columns (case-insensitive search)
            def find_column(df, keywords):
                for col in df.columns:
                    for keyword in keywords:
                        if keyword.lower() in col.lower():
                            return col
                return None
            
            player_col = find_column(shots_df, ['Player'])
            minute_col = find_column(shots_df, ['Minute'])
            squad_col = find_column(shots_df, ['Squad'])
            outcome_col = find_column(shots_df, ['Outcome'])
            xg_col = find_column(shots_df, ['xG'])
            psxg_col = find_column(shots_df, ['PSxG'])
            
            if not all([player_col, minute_col, squad_col, outcome_col, xg_col]):
                print(f"  Missing required columns in shots table")
                print(f"  Available columns: {list(shots_df.columns)}")
                return None
            
            # Process shots data
            shots_df["Event Type"] = shots_df.apply(
                lambda row: "Penalty" if "(pen)" in str(row[player_col]).lower() else "Shot", 
                axis=1
            )
            
            shots_df["Minute"] = shots_df[minute_col].str.extract(r'(\d+)').fillna('0')
            shots_df["Team"] = shots_df[squad_col]
            shots_df["Player"] = shots_df[player_col]
            shots_df["Outcome"] = shots_df[outcome_col]
            shots_df["xG"] = shots_df[xg_col]
            shots_df["PSxG"] = shots_df[psxg_col] if psxg_col else 0
            
            # Select final columns
            final_columns = ["Minute", "Team", "Player", "Event Type", "Outcome", "xG", "PSxG"]
            shots_df = shots_df[final_columns]
            
            # Clean up data
            shots_df = shots_df[shots_df['Minute'] != ''].copy()
            
            # Get red cards data (simplified)
            red_cards_data = []
            events = soup.find_all('div', class_=re.compile(r'^event\s'))
            for event in events:
                event_text = event.get_text().lower()
                if 'red card' in event_text or 'second yellow' in event_text:
                    # Extract basic info - this is simplified
                    player_link = event.find('a')
                    if player_link:
                        player_name = player_link.get_text(strip=True)
                        # Try to extract minute and team
                        time_match = re.search(r"(\d+)'", event.get_text())
                        minute = time_match.group(1) if time_match else '90'
                        
                        red_cards_data.append({
                            'Minute': minute,
                            'Team': 'Unknown',  # Would need more complex logic to determine
                            'Player': player_name,
                            'Event Type': 'Red Card',
                            'Outcome': 'Red Card',
                            'xG': 0,
                            'PSxG': 0
                        })
            
            # Add red cards if any found
            if red_cards_data:
                red_cards_df = pd.DataFrame(red_cards_data)
                shots_df = pd.concat([shots_df, red_cards_df], ignore_index=True)
            
            # Clean and convert data types
            shots_df["Minute"] = pd.to_numeric(shots_df["Minute"], errors='coerce')
            shots_df["xG"] = pd.to_numeric(shots_df["xG"], errors='coerce')
            shots_df["PSxG"] = pd.to_numeric(shots_df["PSxG"], errors='coerce')
            shots_df.fillna(0.00, inplace=True)
            shots_df.sort_values(by=["Minute"], inplace=True)
            
            # Add match metadata
            shots_df["match_url"] = url
            shots_df["match_date"] = match_date
            shots_df["home_team"] = home_team
            shots_df["away_team"] = away_team
            shots_df["division"] = division
            
            # Add season
            if match_date:
                match_datetime = datetime.strptime(match_date, '%Y-%m-%d')
                if match_datetime.month >= 8:
                    shots_df["season"] = match_datetime.year
                else:
                    shots_df["season"] = match_datetime.year - 1
            else:
                try:
                    shots_df["season"] = int(self.season.split("-")[0])
                except:
                    shots_df["season"] = None
                    
            return shots_df.reset_index(drop=True)
            
        except Exception as e:
            print(f"  Error processing shots table: {str(e)}")
            return None

    def scrape_matches(self):
        """Main scraping method with improved error handling"""
        try:
            print(f"\nStarting scrape for season {self.season}, matches from last {self.days_back} days")
            
            # First, get the fixtures page with Selenium
            print("Loading fixtures page...")
            self.driver.get(self.base_url)
            self.smart_delay(3, 5)
            
            # Find the fixtures table
            table = self.find_fixtures_table()
            if not table:
                raise Exception("Could not find fixtures table")
            
            # Get all match report links
            match_links = []
            rows = table.find_elements(By.TAG_NAME, "tr")
            print(f"Found {len(rows)-1} total matches to check")
            
            for row in reversed(rows[1:]):  # Most recent first
                try:
                    # Check date
                    date_cell = row.find_element(By.XPATH, ".//td[@data-stat='date']")
                    match_date_text = date_cell.text.strip()
                    
                    if not match_date_text:
                        continue
                    
                    try:
                        match_date = datetime.strptime(match_date_text, '%Y-%m-%d')
                    except ValueError:
                        try:
                            match_date = datetime.strptime(match_date_text, '%a %d/%m/%Y')
                        except ValueError:
                            continue
                    
                    if match_date < self.cutoff_date:
                        print(f"Reached matches older than {self.days_back} days, stopping search")
                        break
                    
                    # Get match report link
                    try:
                        match_report = row.find_element(By.XPATH, ".//td/a[text()='Match Report']")
                        match_url = match_report.get_attribute('href')
                        match_links.append((match_date_text, match_url))
                    except NoSuchElementException:
                        continue
                        
                except Exception as e:
                    continue
            
            print(f"Found {len(match_links)} matches within date range")
            
            # Now process matches using requests
            matches_collected = 0
            for i, (date_text, url) in enumerate(match_links, 1):
                print(f"\nProcessing match {i}/{len(match_links)} from {date_text}")
                
                match_df = self.get_match_data(url)
                if match_df is not None:
                    self.match_data.append(match_df)
                    matches_collected += 1
                
                # Smart delay between matches
                if i < len(match_links):  # Don't delay after last match
                    self.smart_delay(8, 15)  # Longer delays to avoid detection
            
            print(f"\nProcessed {len(match_links)} matches, collected data for {matches_collected} matches")
            return matches_collected > 0
            
        except Exception as e:
            print(f"An error occurred during scraping: {e}")
            return False

    def find_fixtures_table(self):
        """Find the fixtures table with multiple fallback selectors"""
        try:
            table_selectors = [
                f"table#sched_{self.season}_9_1",
                "table.stats_table.sortable",
                "table[class*='stats_table']",
                "//table[contains(@class, 'stats_table')]"
            ]
            
            for selector in table_selectors:
                try:
                    if selector.startswith("//"):
                        table = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, selector))
                        )
                    else:
                        table = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                    print(f"Found fixtures table using selector: {selector}")
                    return table
                except TimeoutException:
                    continue
                    
            raise Exception("Could not find fixtures table with any selector")
            
        except Exception as e:
            print(f"Error finding fixtures table: {e}")
            return None

    def save_results(self):
        """Save results to CSV and database"""
        if self.match_data:
            combined_df = pd.concat(self.match_data, ignore_index=True)
            
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'recent_matches_{self.days_back}days_{timestamp}.csv'
            
            os.makedirs('data', exist_ok=True)
            filepath = os.path.join('data', filename)
            combined_df.to_csv(filepath, index=False)
            print(f"\n✓ Results saved to {filepath}")
            
            # Save to database
            try:
                conn = sqlite3.connect(self.db_path)
                combined_df.to_sql('prem_data', conn, if_exists='append', index=False)
                print(f"✓ Appended {len(combined_df)} rows to database")
                conn.close()
            except Exception as e:
                print(f"✗ Database error: {str(e)}")
                
            print(f"\n📊 Total events collected: {len(combined_df)}")
            
            # Show summary
            print(f"📋 Summary:")
            print(f"   - Unique matches: {combined_df.groupby(['home_team', 'away_team', 'match_date']).ngroups}")
            print(f"   - Total shots: {len(combined_df[combined_df['Event Type'] == 'Shot'])}")
            print(f"   - Total penalties: {len(combined_df[combined_df['Event Type'] == 'Penalty'])}")
            print(f"   - Red cards: {len(combined_df[combined_df['Event Type'] == 'Red Card'])}")
            
            return True
        else:
            print("\n⚠ No match data collected")
            return False

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()
        if hasattr(self, 'session'):
            self.session.close()
            
    def run(self):
        """Main run method"""
        try:
            if self.scrape_matches():
                self.save_results()
            else:
                print("❌ No data was successfully scraped")
        except KeyboardInterrupt:
            print("\n🛑 Scraping interrupted by user")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
        finally:
            self.cleanup()
            print("\n🏁 Script completed")


if __name__ == "__main__":
    # Configuration
    season = "2024-2025"
    days_back = 10000  # Reduced from 157 to avoid too many requests
    db_path = r"C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db"
    
    scraper = ImprovedMatchDataScraper(
        season=season, 
        days_back=days_back, 
        headless=True,  # Set to False for debugging
        db_path=db_path
    )
    scraper.run()