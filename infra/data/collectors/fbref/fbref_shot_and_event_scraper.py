import pandas as pd
import numpy as np
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
import cloudscraper
from bs4 import BeautifulSoup
import re
import html

# Try to import lxml, use html.parser as fallback
try:
    import lxml
    DEFAULT_PARSER = 'lxml'
    print("✓ Using lxml parser")
except ImportError:
    print("⚠ lxml parser not available, using html.parser instead")
    print("  Recommend: pip install lxml")
    DEFAULT_PARSER = 'html.parser'

def extract_division_from_url(url):
    """
    Extract division/league from the match URL
    
    Args:
        url (str): FBRef match URL
    
    Returns:
        str: Division/League name
    """
    # Common league patterns in FBRef URLs
    league_patterns = {
        'Premier-League': 'Premier League',
        'Championship': 'Championship',
        'League-One': 'League One',
        'League-Two': 'League Two',
        'La-Liga': 'La Liga',
        'Serie-A': 'Serie A',
        'Bundesliga': 'Bundesliga',
        'Ligue-1': 'Ligue 1',
        'Champions-League': 'Champions League',
        'Europa-League': 'Europa League',
        'FA-Cup': 'FA Cup',
        'EFL-Cup': 'EFL Cup',
        'World-Cup': 'World Cup'
    }
    
    # Extract from URL pattern
    for pattern, league_name in league_patterns.items():
        if pattern in url:
            return league_name
    
    # Fallback: try to extract from URL structure
    # FBRef URLs often have format: .../matches/id/teams-date-league
    try:
        parts = url.split('/')
        if len(parts) > 5:
            # Look for league indication in the last part
            last_part = parts[-1]
            if 'Premier-League' in last_part:
                return 'Premier League'
            elif 'Championship' in last_part:
                return 'Championship'
            # Add more patterns as needed
    except:
        pass
    
    return 'Unknown'

def extract_team_names(soup):
    """
    Extract home and away team names from the scorebox
    
    Args:
        soup: BeautifulSoup object
    
    Returns:
        tuple: (home_team, away_team)
    """
    scorebox = soup.find('div', {'class': "scorebox"})
    team_links = scorebox.select('a[href*="/squads/"]')
    
    if len(team_links) >= 2:
        home_team = team_links[0].text.strip()
        away_team = team_links[1].text.strip()
    else:
        home_team, away_team = "Home Team", "Away Team"
    
    return home_team, away_team

def extract_match_events(soup):
    """
    Extract match events (goals, cards, etc.) from the match page
    
    Args:
        soup: BeautifulSoup object
    
    Returns:
        pandas.DataFrame: Match events data
    """
    # Find all event elements using regex to match class names starting with 'event'
    events = soup.find_all('div', class_=re.compile(r'^event [ab]'))
    
    # Initialize lists to hold event data
    times = []
    scores = []
    players = []
    event_types = []
    teams = []
    sides = []

    # Get team names for own goal handling and validation
    home_team, away_team = extract_team_names(soup)

    # Extract data for each event
    for event in events:
        # Extract time and score
        time_score_div = event.find('div')
        if time_score_div:
            time_score_text = html.unescape(time_score_div.get_text(strip=True))
            time = time_score_text.split('’')[0] if '’'in time_score_text else time_score_text
            score = time_score_text.split('’')[1] if '’' in time_score_text else ''
            times.append(time)
            scores.append(score)

        # Extract player name
        player = event.find('a').get_text(strip=True) if event.find('a') else ''
        players.append(player)

        # Extract event type based on class
        event_type = 'Unknown'
        icon_div = event.find('div', class_=re.compile(r'event_icon'))
        if icon_div:
            classes = icon_div.get("class", [])
            specific_classes = [c for c in classes if c != "event_icon"]
            if specific_classes:
                event_type = specific_classes[0].replace("_", " ").title()
        event_types.append(event_type)

        # Extract team name from logo alt text (this is the player's team)
        player_team = ''
        logo_img = event.find('img', class_='teamlogo')
        if logo_img:
            alt_text = logo_img.get('alt', '')
            player_team = alt_text.replace(' Club Crest', '').strip()
        
        # Determine side based on event class
        event_classes = event.get('class', [])
        side = 'unknown'
        
        if 'event' in event_classes and len(event_classes) > 1:
            # Check if this is event a or event b
            # Note: On FBRef, 'a' = away team, 'b' = home team
            if 'a' in event_classes:
                side = 'home'
            elif 'b' in event_classes:
                side = 'away'
        
        # For most events, the team is the player's team
        team_name = player_team
        
        # Handle own goals - for own goals, the benefiting team should be the opponent
        if event_type == 'Own Goal':
            # For own goals, we need to assign the goal to the OPPONENT team
            if side == 'away':
                # Player is from home team, but goal benefits away team
                team_name = away_team
                # Keep the side as 'away' because that's who benefits
                side = 'away'
            elif side == 'home':
                # Player is from away team, but goal benefits home team  
                team_name = home_team
                # Keep the side as 'home' because that's who benefits
                side = 'home'
        
        
        teams.append(team_name)
        sides.append(side)

    # Create DataFrame with standardized column names
    df = pd.DataFrame({
        'minute': times,
        'score': scores,
        'player': players,
        'event_type': event_types,
        'team': teams,
        'side': sides
    })

    return df

def extract_shots_data(soup, home_team):
    """
    Extract shots data from the match page
    
    Args:
        soup: BeautifulSoup object
        home_team (str): Name of the home team
    
    Returns:
        pandas.DataFrame: Shots data
    """
    def normalize_team_name(name):
        """Normalize team names for comparison"""
        # Common team name variations between scorebox and tables
        name = name.strip()
        
        # Remove common suffixes/prefixes
        replacements = {
            "Brighton & Hove Albion": "Brighton",
            "West Ham United": "West Ham",
            "Wolverhampton Wanderers": "Wolves",
            "Tottenham Hotspur": "Tottenham",
            "Nottingham Forest": "Nott'ham Forest",
            "Manchester United": "Manchester Utd",
            "Newcastle United": "Newcastle Utd"
        }
        
        # Check both directions
        for full_name, short_name in replacements.items():
            if name == full_name:
                return short_name
            elif name == short_name:
                return short_name
        
        return name
    
    table = soup.find_all('table')
    
    if len(table) < 18:
        return pd.DataFrame()
    
    # Shots table: Both squads (assuming it's table[17])
    shots_table = table[17]

    # Extract column headers from the second row
    headers = [th.get_text(strip=True) for th in shots_table.find_all('tr')[1].find_all('th')]

    # Initialize list to hold row data
    rows_data = []

    # Extract data from each row
    for tr in shots_table.find_all('tbody')[0].find_all('tr'):
        cols = tr.find_all(['th', 'td'])
        row_data = [col.get_text(strip=True) for col in cols]
        rows_data.append(row_data)

    # Create the DataFrame
    df = pd.DataFrame(rows_data, columns=headers)
    
    # Check if required columns exist
    required_cols = ["Minute", "Player", "Squad", "xG", "PSxG", "Outcome"]
    available_cols = [col for col in required_cols if col in df.columns]
    
    if len(available_cols) < 4:  # At least 4 of the required columns
        return pd.DataFrame()
    
    df = df[available_cols]
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # Standardize column names to lowercase with underscores
    column_mapping = {
        'Minute': 'minute',
        'Player': 'player',
        'Squad': 'squad',
        'xG': 'xg',
        'PSxG': 'psxg',
        'Outcome': 'outcome'
    }
    
    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    if "squad" in df.columns:
        # Normalize team names for comparison
        normalized_home_team = normalize_team_name(home_team)
        df["normalized_squad"] = df["squad"].apply(normalize_team_name)
        
        df["side"] = np.where(df["normalized_squad"] == normalized_home_team, "home", "away")
        
        # Drop the helper column
        df = df.drop("normalized_squad", axis=1)
        
        # Rename squad to team for consistency
        df = df.rename(columns={"squad": "team"})
    
    df = df[df["minute"] != ""]

    return df

def scrape_match_data(url, season=None):
    """
    Main function to scrape match data and return events and shots DataFrames
    
    Args:
        url (str): FBRef match URL
        season (str): Season string (e.g., "2024-2025")
    
    Returns:
        tuple: (match_events_df, shots_df)
    """
    # Setup cloudscraper session
    session = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'mobile': False
        }
    )
    
    session.headers.update({
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    try:
        # Get page content
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, DEFAULT_PARSER)
        
        # Extract team names
        home_team, away_team = extract_team_names(soup)
        
        # Extract division from URL
        division = extract_division_from_url(url)
        
        # Get match events
        match_events = extract_match_events(soup)
        if not match_events.empty:
            match_events['division'] = division
            if season:
                match_events['season'] = season
        
        # Get shots data
        shots = extract_shots_data(soup, home_team)
        if not shots.empty:
            shots['division'] = division
            if season:
                shots['season'] = season
        
        return match_events, shots
        
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        session.close()

class MatchDataScraper:
    def __init__(self, season, days_back=7, headless=True, db_path="team_model_db"):
        self.season = season
        self.days_back = days_back
        self.base_url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
        self.events_data = []  # Store events DataFrames
        self.shots_data = []   # Store shots DataFrames
        self.db_path = db_path
        self.cutoff_date = datetime.now() - timedelta(days=days_back)
        
        self.setup_session()
        self.setup_driver(headless)
        print(f"Scraping matches played after: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
    def setup_session(self):
        """Setup cloudscraper session with proper headers"""
        self.session = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            }
        )
        
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
    def setup_driver(self, headless):
        """Setup Selenium driver with anti-detection measures"""
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        
        # Anti-detection options
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-gpu')
        
        # Set user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        options.add_argument(f'user-agent={user_agent}')
        
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.set_window_size(1920, 1080)
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("Make sure ChromeDriver is installed and in PATH")
            raise

    def smart_delay(self, min_seconds=3, max_seconds=8):
        """Implement smart delays"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

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

    def scrape_matches(self):
        """Main scraping method"""
        try:
            print(f"\nStarting scrape for season {self.season}, matches from last {self.days_back} days")
            
            # Get the fixtures page with Selenium
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
            
            # Process matches using the new scraping functions
            matches_collected = 0
            for i, (date_text, url) in enumerate(match_links, 1):
                print(f"\nProcessing match {i}/{len(match_links)} from {date_text}")
                print(f"  URL: {url}")
                
                match_events, shots = scrape_match_data(url, self.season)
                
                if not match_events.empty or not shots.empty:
                    if not match_events.empty:
                        match_events['match_url'] = url
                        match_events['match_date'] = date_text
                        self.events_data.append(match_events)
                    
                    if not shots.empty:
                        shots['match_url'] = url
                        shots['match_date'] = date_text
                        self.shots_data.append(shots)
                    
                    matches_collected += 1
                    print(f"  Collected {len(match_events)} events and {len(shots)} shots")
                    
                    # Print division info
                    division = extract_division_from_url(url)
                    print(f"  Division: {division}")
                else:
                    print(f"  No data collected")
                
                # Smart delay between matches
                if i < len(match_links):
                    self.smart_delay(5, 10)
            
            print(f"\nProcessed {len(match_links)} matches, collected data for {matches_collected} matches")
            return matches_collected > 0
            
        except Exception as e:
            print(f"An error occurred during scraping: {e}")
            return False

    def test_specific_urls(self, urls):
        """Test scraping specific match URLs"""
        print(f"\nTesting {len(urls)} specific URLs...")
        
        matches_collected = 0
        for i, url in enumerate(urls, 1):
            print(f"\n--- Testing URL {i}/{len(urls)} ---")
            print(f"URL: {url}")
            
            match_events, shots = scrape_match_data(url, self.season)
            
            if not match_events.empty or not shots.empty:
                if not match_events.empty:
                    match_events['match_url'] = url
                    self.events_data.append(match_events)
                
                if not shots.empty:
                    shots['match_url'] = url
                    self.shots_data.append(shots)
                
                matches_collected += 1
                print(f"Successfully collected {len(match_events)} events and {len(shots)} shots")
                
                # Print division info
                division = extract_division_from_url(url)
                print(f"Division: {division}")
            else:
                print(f"Failed to collect data")
            
            # Add delay between URLs
            if i < len(urls):
                self.smart_delay(3, 7)
        
        print(f"\nTest Results: {matches_collected}/{len(urls)} URLs successfully scraped")
        return matches_collected > 0

    def save_results(self):
        """Save results to separate CSV files and database"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('data', exist_ok=True)
        
        events_saved = False
        shots_saved = False
        
        # Save Events data
        if self.events_data:
            events_df = pd.concat(self.events_data, ignore_index=True)
            
            events_filename = f'match_events_{self.days_back}days_{timestamp}.csv'
            events_filepath = os.path.join('data', events_filename)
            events_df.to_csv(events_filepath, index=False)
            print(f"\nEvents data saved to {events_filepath}")
            print(f"Total events: {len(events_df)}")
            
            # Show events summary
            if 'event_type' in events_df.columns:
                event_summary = events_df['event_type'].value_counts()
                print("Events Summary:")
                for event_type, count in event_summary.items():
                    print(f"   - {event_type}: {count}")
            
            # Show division breakdown
            if 'division' in events_df.columns:
                division_summary = events_df['division'].value_counts()
                print("Division Summary (Events):")
                for division, count in division_summary.items():
                    print(f"   - {division}: {count}")
            
            # Show season breakdown
            if 'season' in events_df.columns:
                season_summary = events_df['season'].value_counts()
                print("Season Summary (Events):")
                for season, count in season_summary.items():
                    print(f"   - {season}: {count}")
            
            # Save events to database
            try:
                conn = sqlite3.connect(self.db_path)
                events_df.to_sql('fbref_match_events', conn, if_exists='append', index=False)
                print(f"Appended {len(events_df)} events to database")
                conn.close()
            except Exception as e:
                print(f"Database error (events): {str(e)}")
            
            events_saved = True
        
        # Save Shots data
        if self.shots_data:
            shots_df = pd.concat(self.shots_data, ignore_index=True)
            
            shots_filename = f'match_shots_{self.days_back}days_{timestamp}.csv'
            shots_filepath = os.path.join('data', shots_filename)
            shots_df.to_csv(shots_filepath, index=False)
            print(f"\nShots data saved to {shots_filepath}")
            print(f"Total shots: {len(shots_df)}")
            
            # Show shots summary
            if 'outcome' in shots_df.columns:
                shots_summary = shots_df['outcome'].value_counts()
                print("Shots Summary:")
                for outcome, count in shots_summary.items():
                    print(f"   - {outcome}: {count}")
            
            # Show division breakdown
            if 'division' in shots_df.columns:
                division_summary = shots_df['division'].value_counts()
                print("Division Summary (Shots):")
                for division, count in division_summary.items():
                    print(f"   - {division}: {count}")
            
            # Show season breakdown
            if 'season' in shots_df.columns:
                season_summary = shots_df['season'].value_counts()
                print("Season Summary (Shots):")
                for season, count in season_summary.items():
                    print(f"   - {season}: {count}")
            
            # Save shots to database
            try:
                conn = sqlite3.connect(self.db_path)
                shots_df.to_sql('fbref_match_shots', conn, if_exists='append', index=False)
                print(f"Appended {len(shots_df)} shots to database")
                conn.close()
            except Exception as e:
                print(f"Database error (shots): {str(e)}")
            
            shots_saved = True
        
        if not events_saved and not shots_saved:
            print("\nNo data collected to save")
            return False
        
        return True

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()
        if hasattr(self, 'session'):
            self.session.close()
            
    def run(self, test_urls=None):
        """Main run method with optional URL testing"""
        try:
            if test_urls:
                # Test specific URLs mode
                if self.test_specific_urls(test_urls):
                    self.save_results()
                else:
                    print("No data was successfully scraped from test URLs")
            else:
                # Normal scraping mode
                if self.scrape_matches():
                    self.save_results()
                else:
                    print("No data was successfully scraped")
        except KeyboardInterrupt:
            print("\nScraping interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        finally:
            self.cleanup()
            print("\nScript completed")


if __name__ == "__main__":
    # Configuration
    season = "2024-2025"
    days_back = 10000
    db_path = r"C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db"
    
    # Test URLs
    test_urls = [
        "https://fbref.com/en/matches/e4bb1c35/Tottenham-Hotspur-Brighton-and-Hove-Albion-May-25-2025-Premier-League",
        "https://fbref.com/en/matches/ee9ce5e2/North-London-Derby-Arsenal-Tottenham-Hotspur-January-15-2025-Premier-League",
        "https://fbref.com/en/matches/157740ee/Chelsea-Liverpool-May-4-2025-Premier-League,2025-05-04"
    ]
    
    scraper = MatchDataScraper(
        season=season, 
        days_back=days_back, 
        headless=True,
        db_path=db_path
    )
    
    # To test specific URLs
    #scraper.run(test_urls=test_urls)
    
    # For normal scraping
    scraper.run()

