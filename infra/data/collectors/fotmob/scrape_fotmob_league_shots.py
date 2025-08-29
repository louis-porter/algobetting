import requests
from bs4 import BeautifulSoup
import time
import csv
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime
import concurrent.futures
from threading import Lock
import json

# Global variables for thread safety
csv_lock = Lock()
driver_pool = []

def get_optimized_chrome_options():
    """Get optimized Chrome options for faster scraping"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-plugins')
    chrome_options.add_argument('--disable-images')  # Don't load images for speed
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--disable-features=VizDisplayCompositor')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    
    # Remove problematic options that can cause startup issues
    # chrome_options.add_argument('--disable-javascript')  # REMOVED - needed for dynamic content
    # chrome_options.add_argument('--disable-css')  # REMOVED - needed for selectors
    # chrome_options.add_argument('--page-load-strategy=eager')  # REMOVED - can cause issues
    
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Add user agent to avoid detection
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    return chrome_options

def create_driver():
    """Create a new Chrome driver instance"""
    chrome_options = get_optimized_chrome_options()
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Set reasonable timeouts
        driver.set_page_load_timeout(30)  # Increased from 15
        driver.implicitly_wait(5)  # Increased from 2
        
        return driver
    
    except Exception as e:
        print(f"Error creating driver: {e}")
        # Fallback with minimal options if the optimized version fails
        print("Trying with minimal Chrome options...")
        
        fallback_options = Options()
        fallback_options.add_argument('--headless')
        fallback_options.add_argument('--no-sandbox')
        fallback_options.add_argument('--disable-dev-shm-usage')
        
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=fallback_options)
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(5)
            return driver
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise

def parse_date_to_iso(date_string, default_year=None):
    """Convert various date formats to YYYY-MM-DD format (unchanged)"""
    if not date_string or date_string.strip() == '':
        return None
    
    date_string = date_string.strip()
    
    try:
        date_parts = date_string.split()
        
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if len(date_parts) > 0 and date_parts[0].lower() in days_of_week:
            date_parts = date_parts[1:]
        
        year = None
        day = None
        month = None
        
        for i, part in enumerate(date_parts):
            if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2100:
                year = int(part)
                date_parts = date_parts[:i] + date_parts[i+1:]
                break
        
        if year is None:
            if default_year:
                year = default_year
            else:
                year = datetime.now().year
        
        if len(date_parts) >= 2:
            day = date_parts[0]
            month = date_parts[1]
            
            month_map = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
                'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
                'july': 7, 'jul': 7, 'august': 8, 'aug': 8,
                'september': 9, 'sep': 9, 'sept': 9, 'october': 10, 'oct': 10,
                'november': 11, 'nov': 11, 'december': 12, 'dec': 12
            }
            
            month_num = month_map.get(month.lower())
            if month_num and day.isdigit():
                formatted_date = f"{year}-{month_num:02d}-{int(day):02d}"
                return formatted_date
    
    except Exception as e:
        print(f"Error parsing date '{date_string}': {e}")
    
    return date_string

def scrape_league_rounds_fast(base_url, max_rounds=30, season_year=None):
    """Optimized version of round scraping"""
    driver = create_driver()
    all_matches = []
    
    try:
        # Pre-compile regex patterns
        team_id_pattern = re.compile(r'teamlogo/(\d+)_')
        
        for round_num in range(max_rounds):
            print(f"Scraping round {round_num + 1}...")
            
            round_url = base_url.replace('round=0', f'round={round_num}')
            
            try:
                driver.get(round_url)
                
                # Wait for matches to load with timeout
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'a.css-1ajdexg-MatchWrapper'))
                    )
                except TimeoutException:
                    print(f"No matches found in round {round_num + 1}, stopping...")
                    break
                
                # Get all matches at once
                match_elements = driver.find_elements(By.CSS_SELECTOR, 'a.css-1ajdexg-MatchWrapper')
                if not match_elements:
                    break
                
                # Extract all data in batch
                current_date = None
                for match_element in match_elements:
                    try:
                        # Get match date more efficiently
                        match_date = driver.execute_script("""
                            var match = arguments[0];
                            var current = match;
                            
                            while (current && current !== document.body) {
                                var prev = current.previousElementSibling;
                                while (prev) {
                                    if (prev.tagName === 'H3' && prev.textContent) {
                                        return prev.textContent.trim();
                                    }
                                    prev = prev.previousElementSibling;
                                }
                                current = current.parentElement;
                            }
                            return null;
                        """, match_element) or current_date
                        
                        current_date = match_date
                        iso_date = parse_date_to_iso(match_date, season_year)
                        
                        # Extract all match data at once
                        match_data = driver.execute_script("""
                            var match = arguments[0];
                            var data = {};
                            
                            // Get URL
                            data.url = match.href;
                            
                            // Get teams
                            var teams = match.querySelectorAll('.css-1o142s8-TeamName');
                            data.homeTeam = teams[0] ? teams[0].textContent.trim() : 'Unknown';
                            data.awayTeam = teams[1] ? teams[1].textContent.trim() : 'Unknown';
                            
                            // Get score
                            var scoreEl = match.querySelector('.css-1wwsq70-LSMatchStatusScore');
                            data.score = scoreEl ? scoreEl.textContent.trim() : 'N/A';
                            
                            // Get status
                            var statusEl = match.querySelector('.css-1t50dhw-StatusDotCSS') || 
                                          match.querySelector('.css-1ubkvjq-LSMatchStatusReason');
                            data.status = statusEl ? (statusEl.title || statusEl.textContent.trim()) : 'Unknown';
                            
                            return data;
                        """, match_element)
                        
                        if match_data['url']:
                            match_url = match_data['url'] if match_data['url'].startswith('http') else f"https://www.fotmob.com{match_data['url']}"
                            
                            all_matches.append({
                                'round': round_num + 1,
                                'date': iso_date,
                                'home_team': match_data['homeTeam'],
                                'away_team': match_data['awayTeam'],
                                'score': match_data['score'],
                                'status': match_data['status'],
                                'match_url': match_url
                            })
                    
                    except Exception as e:
                        print(f"Error processing match: {e}")
                        continue
                
                print(f"Found {len([m for m in all_matches if m['round'] == round_num + 1])} matches in round {round_num + 1}")
                
            except Exception as e:
                print(f"Error accessing round {round_num + 1}: {e}")
                continue
        
        return all_matches
        
    finally:
        driver.quit()

def scrape_match_xg_optimized(match_data):
    """Optimized xG scraping for a single match"""
    match_url, match_date, home_team, away_team, round_num = match_data
    
    # Ensure stats tab
    if ':tab=stats' not in match_url:
        match_url = match_url + ('#:tab=stats' if '#' not in match_url else ':tab=stats')
    
    driver = create_driver()
    
    try:
        driver.get(match_url)
        
        # Wait for stats to load
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.css-g9mdo5-XGItemValue'))
            )
        except TimeoutException:
            print(f"Stats not loaded for {home_team} vs {away_team}")
            return []
        
        # Get team mapping once - completely rewritten approach
        team_id_mapping = {}
        try:
            # Get team IDs and names more reliably
            team_data = driver.execute_script("""
                var teams = [];
                
                // Method 1: Try to get from match header
                var headerTeams = document.querySelectorAll('[class*="TeamName"], [class*="team-name"]');
                for (var team of headerTeams) {
                    var text = team.textContent.trim();
                    if (text && text.length > 2 && !text.match(/^\\d+[:\\-\\s]*\\d*$/)) {
                        // Clean up team name
                        text = text.replace(/\\d+/g, '').replace(/['"]/g, '').trim();
                        if (text.length > 2) {
                            teams.push(text);
                        }
                    }
                }
                
                // Method 2: Extract from URL or page title if header method failed
                if (teams.length < 2) {
                    teams = [];
                    var title = document.title;
                    var urlMatch = window.location.href.match(/matches\\/([^-]+)-vs-([^/]+)/);
                    if (urlMatch) {
                        teams.push(urlMatch[1].replace(/-/g, ' '));
                        teams.push(urlMatch[2].replace(/-/g, ' '));
                    }
                }
                
                // Get team IDs from logos
                var teamIds = [];
                var logos = document.querySelectorAll('img[src*="teamlogo"]');
                var processed = new Set();
                
                for (var logo of logos) {
                    if (logo.src.includes('_small.png') || logo.src.includes('_medium.png')) {
                        var match = logo.src.match(/teamlogo\\/(\\d+)_/);
                        if (match && !processed.has(match[1])) {
                            processed.add(match[1]);
                            teamIds.push(match[1]);
                        }
                    }
                }
                
                return {
                    names: teams.slice(0, 2),
                    ids: teamIds.slice(0, 2)
                };
            """)
            
            # Create mapping
            if team_data['names'] and team_data['ids'] and len(team_data['names']) >= 2 and len(team_data['ids']) >= 2:
                team_id_mapping[team_data['ids'][0]] = team_data['names'][0]
                team_id_mapping[team_data['ids'][1]] = team_data['names'][1]
                print(f"Team mapping: {team_id_mapping}")
            
        except Exception as e:
            print(f"Error getting team mapping: {e}")
            
        # Fallback: Just use the provided team names with IDs we find
        if not team_id_mapping:
            try:
                team_ids = driver.execute_script("""
                    var ids = [];
                    var logos = document.querySelectorAll('img[src*="teamlogo"]');
                    var processed = new Set();
                    
                    for (var logo of logos) {
                        if (logo.src.includes('_small.png') || logo.src.includes('_medium.png')) {
                            var match = logo.src.match(/teamlogo\\/(\\d+)_/);
                            if (match && !processed.has(match[1])) {
                                processed.add(match[1]);
                                ids.push(match[1]);
                            }
                        }
                    }
                    return ids.slice(0, 2);
                """)
                
                if len(team_ids) >= 2:
                    team_id_mapping[team_ids[0]] = home_team
                    team_id_mapping[team_ids[1]] = away_team
                    print(f"Fallback team mapping: {team_id_mapping}")
                    
            except:
                print("Could not establish team mapping - will use player-based detection")
        
        all_shots = []
        shot_count = 0
        max_shots = 50  # Reduced limit
        seen_shots = set()  # Track duplicates with better key
        consecutive_failures = 0
        
        # Keep track of which players belong to which team for better assignment
        player_team_mapping = {}
        
        while shot_count < max_shots and consecutive_failures < 3:
            try:
                # Get all shot data at once with JavaScript
                shot_data = driver.execute_script("""
                    var data = {};
                    
                    // Get xG values
                    var xgElements = document.querySelectorAll('.css-g9mdo5-XGItemValue span');
                    var xgValues = [];
                    for (var el of xgElements) {
                        var text = el.textContent.trim();
                        if (text && text.includes('.')) {
                            try {
                                xgValues.push(parseFloat(text));
                            } catch(e) {}
                        }
                    }
                    data.xg = xgValues[0] || null;
                    data.xgot = xgValues[1] || null;
                    
                    // Get shot result
                    try {
                        var shotInfo = document.querySelector('.css-1jl5w6u-ShotInfo');
                        var spans = shotInfo.querySelectorAll('span');
                        data.result = 'Unknown';
                        for (var i = 0; i < spans.length - 1; i++) {
                            if (spans[i].textContent.trim() === 'Result') {
                                data.result = spans[i + 1].textContent.trim();
                                break;
                            }
                        }
                    } catch(e) {
                        data.result = 'Unknown';
                    }
                    
                    // Get player info
                    try {
                        var playerContainer = document.querySelector('.css-108cnc0-PlayerNameContainer');
                        if (!playerContainer) {
                            throw new Error('No player container found');
                        }
                        
                        var playerText = playerContainer.textContent.trim();
                        
                        // Extract minute more carefully
                        var minuteMatch = playerText.match(/(\\d+)(?:\\s*\\+\\s*(\\d+))?'/);
                        if (minuteMatch) {
                            var baseMinute = parseInt(minuteMatch[1]);
                            var addedTime = minuteMatch[2] ? parseInt(minuteMatch[2]) : 0;
                            data.minute = baseMinute + addedTime;
                        } else {
                            data.minute = null;
                        }
                        
                        // Extract player name more carefully
                        var lines = playerText.split('\\n').filter(line => line.trim().length > 0);
                        if (lines.length > 1) {
                            // Take the last line that's not just a time
                            for (var i = lines.length - 1; i >= 0; i--) {
                                var line = lines[i].trim();
                                if (!line.match(/^\\d+'?\\s*\\+?\\s*\\d*'?\\s*$/)) {
                                    data.playerName = line;
                                    break;
                                }
                            }
                        }
                        
                        if (!data.playerName) {
                            // Remove time info and get what's left
                            data.playerName = playerText.replace(/\\d+'?\\s*\\+?\\s*\\d*'?\\s*/g, '').trim();
                        }
                        
                        // Get team ID
                        var teamImg = playerContainer.querySelector('img.TeamIcon');
                        if (teamImg && teamImg.src) {
                            var teamMatch = teamImg.src.match(/teamlogo\\/(\\d+)_/);
                            data.teamId = teamMatch ? teamMatch[1] : null;
                        } else {
                            data.teamId = null;
                        }
                        
                    } catch(e) {
                        console.log('Error extracting player info:', e);
                        data.playerName = 'Unknown Player';
                        data.teamId = null;
                        data.minute = null;
                    }
                    
                    return data;
                """, shot_count)
                
                # Create stronger unique identifier for duplicate detection
                shot_identifier = f"{shot_data.get('playerName', '')}-{shot_data.get('minute', 0)}-{shot_data.get('xg', 0)}-{shot_data.get('xgot', 0)}-{shot_data.get('result', '')}"
                
                # Check if we've seen this exact shot before
                if shot_identifier in seen_shots:
                    consecutive_failures += 1
                    print(f"Duplicate shot detected: {shot_data.get('playerName', 'Unknown')} at {shot_data.get('minute', '?')}'")
                else:
                    consecutive_failures = 0
                    seen_shots.add(shot_identifier)
                    
                    if shot_data.get('xg') is not None:
                        # Determine team name with improved logic
                        team_id = shot_data.get('teamId')
                        team_name = None
                        player_name = shot_data.get('playerName', 'Unknown')
                        
                        # Try team ID mapping first
                        if team_id and team_id in team_id_mapping:
                            team_name = team_id_mapping[team_id]
                        
                        # If no team mapping or mapping failed, use player-based detection
                        if not team_name or team_name.startswith('Team '):
                            # Check if we've seen this player before
                            if player_name in player_team_mapping:
                                team_name = player_team_mapping[player_name]
                            else:
                                # Use knowledge of common player names to assign teams
                                # This is a heuristic but better than random assignment
                                home_indicators = [home_team.lower().replace(' ', ''), home_team.split()[0].lower()]
                                away_indicators = [away_team.lower().replace(' ', ''), away_team.split()[0].lower()]
                                
                                # Check if player name suggests a team (very basic heuristic)
                                player_lower = player_name.lower()
                                
                                # For now, alternate between teams but keep track
                                existing_home = len([s for s in all_shots if s['team_name'] == home_team])
                                existing_away = len([s for s in all_shots if s['team_name'] == away_team])
                                
                                # Assign to team with fewer shots to try to balance
                                if existing_home <= existing_away:
                                    team_name = home_team
                                else:
                                    team_name = away_team
                                
                                # Remember this player's team for consistency
                                player_team_mapping[player_name] = team_name
                        
                        all_shots.append({
                            'round': round_num,
                            'match_date': match_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'match_url': match_url,
                            'player_name': player_name,
                            'team_name': team_name,
                            'minute': shot_data.get('minute'),
                            'xg': shot_data.get('xg'),
                            'xgot': shot_data.get('xgot'),
                            'result': shot_data.get('result', 'Unknown')
                        })
                
                # Try to navigate to next shot
                try:
                    next_buttons = driver.find_elements(By.CSS_SELECTOR, '.css-65dgen-DropdownBrowseButton')
                    if len(next_buttons) >= 2:
                        next_button = next_buttons[1]
                        if next_button.is_enabled() and next_button.is_displayed():
                            # Double check the button isn't disabled
                            button_classes = next_button.get_attribute('class') or ''
                            button_disabled = next_button.get_attribute('disabled')
                            
                            if 'disabled' not in button_classes.lower() and not button_disabled:
                                driver.execute_script("arguments[0].click();", next_button)
                                time.sleep(1.2)  # Slightly longer wait
                                shot_count += 1
                            else:
                                print("Next button is disabled - reached end")
                                break
                        else:
                            print("Next button not enabled or visible")
                            break
                    else:
                        print("No next button found")
                        break
                        
                except Exception as nav_error:
                    print(f"Navigation error: {nav_error}")
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
                    
            except Exception as e:
                print(f"Error getting shot {shot_count + 1}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("Too many consecutive failures, stopping")
                    break
                shot_count += 1  # Still increment to avoid infinite loop
        
        return all_shots
        
    except Exception as e:
        print(f"Error scraping {home_team} vs {away_team}: {e}")
        return []
    finally:
        driver.quit()

def scrape_full_league_xg_parallel(base_url, max_rounds=30, max_workers=4, season_year=None):
    """Main function with parallel processing"""
    print("Step 1: Getting all match URLs...")
    matches = scrape_league_rounds_fast(base_url, max_rounds, season_year)
    
    if not matches:
        print("No matches found!")
        return [], []
    
    # Save matches summary
    save_matches_summary_to_csv(matches, 'league_matches_summary.csv')
    
    # Filter to finished matches only
    finished_matches = [m for m in matches if m['status'] in ['Full time', 'FT'] and m['score'] != 'N/A']
    print(f"Found {len(finished_matches)} finished matches to process")
    
    if not finished_matches:
        return [], matches
    
    # Prepare match data for parallel processing
    match_data_list = [
        (match['match_url'], match['date'], match['home_team'], match['away_team'], match['round'])
        for match in finished_matches
    ]
    
    all_shots = []
    
    # Process matches in parallel
    print(f"Step 2: Processing matches with {max_workers} parallel workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_match = {
            executor.submit(scrape_match_xg_optimized, match_data): match_data 
            for match_data in match_data_list
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_match):
            match_data = future_to_match[future]
            completed += 1
            
            try:
                shots = future.result(timeout=60)  # 60 second timeout per match
                if shots:
                    all_shots.extend(shots)
                    print(f"[{completed}/{len(match_data_list)}] {match_data[2]} vs {match_data[3]} - {len(shots)} shots")
                else:
                    print(f"[{completed}/{len(match_data_list)}] {match_data[2]} vs {match_data[3]} - No shots found")
                    
            except concurrent.futures.TimeoutError:
                print(f"[{completed}/{len(match_data_list)}] {match_data[2]} vs {match_data[3]} - TIMEOUT")
            except Exception as e:
                print(f"[{completed}/{len(match_data_list)}] {match_data[2]} vs {match_data[3]} - ERROR: {e}")
    
    print(f"\nCompleted! Total shots collected: {len(all_shots)}")
    return all_shots, matches

def save_shots_to_csv(shots, filename='league_xg_data.csv'):
    """Save shot data to CSV (thread-safe)"""
    if not shots:
        print("No shot data to save")
        return
    
    with csv_lock:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['round', 'match_date', 'home_team', 'away_team', 'match_url', 
                         'player_name', 'team_name', 'minute', 'xg', 'xgot', 'result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for shot in shots:
                if isinstance(shot, dict):
                    writer.writerow(shot)
    
    print(f"Shot data saved to {filename}")

def save_matches_summary_to_csv(matches, filename='league_matches_summary.csv'):
    """Save match summary (unchanged)"""
    if not matches:
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['round', 'match_date', 'home_team', 'home_goals', 'away_team', 'away_goals', 'match_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for match in matches:
            home_goals = away_goals = None
            
            if match['score'] != 'N/A' and ' - ' in match['score']:
                try:
                    score_parts = match['score'].split(' - ')
                    if len(score_parts) == 2:
                        home_goals = int(score_parts[0].strip())
                        away_goals = int(score_parts[1].strip())
                except ValueError:
                    pass
            
            writer.writerow({
                'round': match['round'],
                'match_date': match['date'],
                'home_team': match['home_team'],
                'home_goals': home_goals,
                'away_team': match['away_team'],
                'away_goals': away_goals,
                'match_url': match['match_url']
            })

# Usage
if __name__ == "__main__":
    base_url = "https://www.fotmob.com/en-GB/leagues/47/matches/premier-league?season=2024-2025&group=by-round&round=0"
    
    # Use optimized parallel version
    # Adjust max_workers based on your system (2-8 is usually good)
    all_shots, matches = scrape_full_league_xg_parallel(
        base_url, 
        max_rounds=38, 
        max_workers=5,  # Reduce if you get blocked
        season_year=2024
    )
    
    if all_shots:
        save_shots_to_csv(all_shots, 'prem_complete_xg_data_fast.csv')
        
        matches_processed = len(set(shot['match_url'] for shot in all_shots))
        rounds_processed = len(set(shot['round'] for shot in all_shots))
        
        print(f"\n=== SUMMARY ===")
        print(f"Rounds processed: {rounds_processed}")
        print(f"Matches processed: {matches_processed}")
        print(f"Total shots: {len(all_shots)}")
        print(f"Data saved to: prem_complete_xg_data_fast.csv")
    else:
        print("No data collected")