import pandas as pd
import sqlite3
import time
import random
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re

# Try to import lxml, use html.parser as fallback
try:
    import lxml
    DEFAULT_PARSER = 'lxml'
except ImportError:
    print("lxml parser not available, using html.parser instead")
    DEFAULT_PARSER = 'html.parser'

class OwnGoalsUpdater:
    def __init__(self, db_path, table_name, batch_size=50, headless=True):
        """
        Initialize the updater with database path and table name
        
        Args:
            db_path: Path to the SQLite database
            table_name: Name of the table containing match data
            batch_size: Number of matches to process in a batch before committing
            headless: Whether to run the browser in headless mode
        """
        self.db_path = db_path
        self.table_name = table_name
        self.batch_size = batch_size
        self.setup_driver(headless)
        self.matches_processed = 0
        self.matches_updated = 0
        
    def setup_driver(self, headless):
        """Set up the Selenium WebDriver"""
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.maximize_window()

    def random_delay(self, min_seconds=1, max_seconds=3):
        """Add a random delay to avoid detection"""
        time.sleep(random.uniform(min_seconds, max_seconds))

    def get_match_urls(self):
        """Get all unique match URLs from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f'''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self.table_name}'
            ''')
            
            if not cursor.fetchone():
                print(f"Table {self.table_name} doesn't exist!")
                conn.close()
                return []
            
            # Get distinct match URLs
            cursor.execute(f"SELECT DISTINCT match_url FROM {self.table_name}")
            urls = [row[0] for row in cursor.fetchall() if row[0]]
            
            conn.close()
            print(f"Found {len(urls)} unique match URLs in the database")
            return urls
            
        except Exception as e:
            print(f"Error fetching match URLs: {e}")
            return []

    def extract_own_goals(self, url):
        """Extract own goals for home and away teams from a match page"""
        try:
            print(f"Fetching own goals data from: {url}")
            self.driver.get(url)
            self.random_delay(1, 2)  # Short delay to ensure page loads
            
            # Get the page source and parse it with BeautifulSoup
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, DEFAULT_PARSER)
            
            # Extract teams
            team_stats = soup.find('div', id='team_stats_extra')
            if not team_stats:
                print("Team stats section not found")
                return None, None, None, None
                
            teams = team_stats.find_all('div', class_='th')
            teams = [t.text.strip() for t in teams if t.text.strip() != '']
            teams = list(dict.fromkeys(teams))  # Remove duplicates while preserving order
            
            if len(teams) < 2:
                print("Could not identify both teams")
                return None, None, None, None
                
            home_team = teams[0]
            away_team = teams[1]
            
            # Find the misc tables for both teams
            misc_pattern = re.compile(r'stats_[a-z0-9]+_misc')
            misc_tables = soup.find_all('table', id=misc_pattern)
            
            home_team_table = None
            away_team_table = None
            
            # Assign tables to respective teams
            for table in misc_tables:
                caption = table.find("caption")
                if caption and caption.text:
                    caption_text = caption.text.strip()
                    
                    if home_team and home_team in caption_text:
                        home_team_table = table
                    elif away_team and away_team in caption_text:
                        away_team_table = table
            
            # Extract own goals for home team
            home_og = 0
            if home_team_table:
                og_cell = home_team_table.find('tfoot').find('td', {'data-stat': 'own_goals'})
                if og_cell:
                    home_og_text = og_cell.get_text(strip=True)
                    try:
                        home_og = int(home_og_text) if home_og_text else 0
                    except ValueError:
                        home_og = 0
            
            # Extract own goals for away team
            away_og = 0
            if away_team_table:
                og_cell = away_team_table.find('tfoot').find('td', {'data-stat': 'own_goals'})
                if og_cell:
                    away_og_text = og_cell.get_text(strip=True)
                    try:
                        away_og = int(away_og_text) if away_og_text else 0
                    except ValueError:
                        away_og = 0
            
            return home_team, away_team, home_og, away_og
            
        except Exception as e:
            print(f"Error extracting own goals: {e}")
            return None, None, None, None

    def update_database(self, match_url, home_team, away_team, home_og, away_og):
        """Update the goals count in the database to include own goals"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if we have goals column
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'goals' not in columns:
                print("No 'goals' column found in the table - cannot update")
                conn.close()
                return False
            
            # Get the current goals values for both teams
            cursor.execute(f"""
                SELECT team, goals, is_home
                FROM {self.table_name}
                WHERE match_url = ?
            """, (match_url,))
            
            rows = cursor.fetchall()
            if len(rows) != 2:
                print(f"Expected 2 rows for match {match_url}, but found {len(rows)}")
                conn.close()
                return False
            
            updated = False
            for row in rows:
                team, current_goals, is_home = row
                
                # Convert goals to integer
                try:
                    current_goals = int(current_goals) if current_goals else 0
                except ValueError:
                    try:
                        current_goals = float(current_goals) if current_goals else 0
                        current_goals = int(current_goals)
                    except ValueError:
                        print(f"Could not convert goals value '{current_goals}' to a number")
                        current_goals = 0
                
                # Determine if we need to add own goals from the opposing team
                if (is_home == 1 and team == home_team) or (is_home == 'True' and team == home_team):
                    # Home team's goals include away team's own goals
                    if away_og > 0:
                        new_goals = current_goals + away_og
                        cursor.execute(f"""
                            UPDATE {self.table_name}
                            SET goals = ?
                            WHERE match_url = ? AND team = ? AND (is_home = 1 OR is_home = 'True')
                        """, (str(new_goals), match_url, team))
                        print(f"Updated {team} goals: {current_goals} → {new_goals} (added {away_og} own goals from opponent)")
                        updated = True
                elif (is_home == 0 and team == away_team) or (is_home == 'False' and team == away_team):
                    # Away team's goals include home team's own goals
                    if home_og > 0:
                        new_goals = current_goals + home_og
                        cursor.execute(f"""
                            UPDATE {self.table_name}
                            SET goals = ?
                            WHERE match_url = ? AND team = ? AND (is_home = 0 OR is_home = 'False')
                        """, (str(new_goals), match_url, team))
                        print(f"Updated {team} goals: {current_goals} → {new_goals} (added {home_og} own goals from opponent)")
                        updated = True
            
            conn.commit()
            conn.close()
            return updated
            
        except Exception as e:
            print(f"Error updating database: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_own_goals(self):
        """Process all match URLs and update own goals data"""
        urls = self.get_match_urls()
        if not urls:
            print("No match URLs found to process")
            return
        
        batch_count = 0
        try:
            total_matches = len(urls)
            for i, url in enumerate(urls):
                self.matches_processed += 1
                print(f"\nProcessing match {self.matches_processed}/{total_matches}: {url}")
                
                # Extract own goals data
                home_team, away_team, home_og, away_og = self.extract_own_goals(url)
                
                if home_team is None or away_team is None:
                    print("Could not extract team data, skipping match")
                    continue
                
                # Log the own goals found
                print(f"Found own goals: {home_team} (OG: {home_og}), {away_team} (OG: {away_og})")
                
                # Update the database if there are own goals
                if home_og > 0 or away_og > 0:
                    updated = self.update_database(url, home_team, away_team, home_og, away_og)
                    if updated:
                        self.matches_updated += 1
                        batch_count += 1
                
                # Add a delay between requests
                self.random_delay()
                
                # Print progress update
                if (i + 1) % 10 == 0 or (i + 1) == total_matches:
                    completion = (i + 1) / total_matches * 100
                    print(f"Progress: {i + 1}/{total_matches} ({completion:.1f}%)")
                    print(f"Updated {self.matches_updated} matches so far")
                
                # Commit in batches to avoid keeping the database locked for too long
                if batch_count >= self.batch_size:
                    print(f"Processed batch of {batch_count} updates")
                    batch_count = 0
                    
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        except Exception as e:
            print(f"Error processing matches: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            
        print(f"\nUpdating complete!")
        print(f"Processed {self.matches_processed} matches")
        print(f"Updated {self.matches_updated} matches with own goals data")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()
            print("WebDriver closed")


class GoalConsistencyFixer:
    def __init__(self, db_path, table_name):
        """
        Initialize the goals consistency fixer.
        
        Args:
            db_path: Path to the SQLite database
            table_name: Name of the table containing match data
        """
        self.db_path = db_path
        self.table_name = table_name
        self.matches_processed = 0
        self.matches_fixed = 0
    
    def get_match_urls(self):
        """Get all unique match URLs from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f'''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self.table_name}'
            ''')
            
            if not cursor.fetchone():
                print(f"Table {self.table_name} doesn't exist!")
                conn.close()
                return []
            
            # Get distinct match URLs
            cursor.execute(f"SELECT DISTINCT match_url FROM {self.table_name}")
            urls = [row[0] for row in cursor.fetchall() if row[0]]
            
            conn.close()
            print(f"Found {len(urls)} unique match URLs in the database")
            return urls
            
        except Exception as e:
            print(f"Error fetching match URLs: {e}")
            return []
    
    def fix_goal_consistency(self):
        """
        Fix the consistency between goals and opp_goals for each match.
        For each pair of rows with the same match_url, ensure opp_goals equals
        the corresponding team's goals.
        """
        urls = self.get_match_urls()
        if not urls:
            print("No match URLs found to process")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            total_matches = len(urls)
            fixed_matches = 0
            
            for i, url in enumerate(urls):
                self.matches_processed += 1
                
                # Get the two rows for this match
                query = f"""
                    SELECT team, is_home, goals, opp_goals, opp_team
                    FROM {self.table_name}
                    WHERE match_url = ?
                """
                
                match_df = pd.read_sql_query(query, conn, params=(url,))
                
                # Skip if we don't have exactly 2 rows
                if len(match_df) != 2:
                    print(f"Warning: Match {url} has {len(match_df)} rows instead of 2, skipping")
                    continue
                
                # Extract the two rows
                row1 = match_df.iloc[0]
                row2 = match_df.iloc[1]
                
                # Check if team1's goals match team2's opp_goals
                team1_goals = self.safe_convert_to_float(row1['goals'])
                team2_opp_goals = self.safe_convert_to_float(row2['opp_goals'])
                
                # Check if team2's goals match team1's opp_goals
                team2_goals = self.safe_convert_to_float(row2['goals'])
                team1_opp_goals = self.safe_convert_to_float(row1['opp_goals'])
                
                need_to_fix = False
                fixes = []
                
                # Check for discrepancies
                if team1_goals != team2_opp_goals:
                    fixes.append(f"Team2 ({row2['team']}) opp_goals {team2_opp_goals} → {team1_goals}")
                    need_to_fix = True
                
                if team2_goals != team1_opp_goals:
                    fixes.append(f"Team1 ({row1['team']}) opp_goals {team1_opp_goals} → {team2_goals}")
                    need_to_fix = True
                
                # Fix the discrepancies
                if need_to_fix:
                    fixed_matches += 1
                    
                    # Update team2's opp_goals to match team1's goals
                    cursor = conn.cursor()
                    cursor.execute(f"""
                        UPDATE {self.table_name}
                        SET opp_goals = ?
                        WHERE match_url = ? AND team = ?
                    """, (str(team1_goals), url, row2['team']))
                    
                    # Update team1's opp_goals to match team2's goals
                    cursor.execute(f"""
                        UPDATE {self.table_name}
                        SET opp_goals = ?
                        WHERE match_url = ? AND team = ?
                    """, (str(team2_goals), url, row1['team']))
                    
                    conn.commit()
                    
                    print(f"Fixed match {url} - {' / '.join(fixes)}")
                
                # Print progress update
                if (i + 1) % 100 == 0 or (i + 1) == total_matches:
                    completion = (i + 1) / total_matches * 100
                    print(f"Progress: {i + 1}/{total_matches} ({completion:.1f}%)")
                    print(f"Fixed {fixed_matches} matches so far")
            
            self.matches_fixed = fixed_matches
            
        except Exception as e:
            print(f"Error processing matches: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
            
        print(f"\nConsistency fix complete!")
        print(f"Processed {self.matches_processed} matches")
        print(f"Fixed {self.matches_fixed} matches with inconsistent goal data")
    
    def safe_convert_to_float(self, value):
        """Safely convert a value to float, handling various formats"""
        if value is None:
            return 0.0
        
        try:
            return float(value)
        except (ValueError, TypeError):
            # Try to handle format variations
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point
                clean_value = ''.join(c for c in value if c.isdigit() or c == '.')
                try:
                    return float(clean_value) if clean_value else 0.0
                except ValueError:
                    return 0.0
            return 0.0


if __name__ == "__main__":
    # Set the database path and table name
    db_path = r"C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db"  # Update this path
    table_name = "fbref_match_summary"  # Update this with your table name
    #batch_size = 50  # Update in batches to avoid long locks
    
    # Run the updater
    #updater = OwnGoalsUpdater(db_path=db_path, table_name=table_name, batch_size=batch_size, headless=True)
    #print(f"Starting own goals update for database: {db_path}, table: {table_name}")
    #updater.update_own_goals()

    # Run the fixer
    fixer = GoalConsistencyFixer(db_path=db_path, table_name=table_name)
    print(f"Starting goal consistency fix for database: {db_path}, table: {table_name}")
    fixer.fix_goal_consistency()