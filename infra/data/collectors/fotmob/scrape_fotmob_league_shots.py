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
from datetime import datetime

def parse_date_to_iso(date_string, default_year=None):
    """
    Convert various date formats to YYYY-MM-DD format
    Extracts year from the date string if present, otherwise uses current year or default_year
    """
    if not date_string or date_string.strip() == '':
        return None
    
    date_string = date_string.strip()
    
    try:
        # Handle formats like "Sunday 25 May 2024", "Monday 1 June", "25 May 2023", etc.
        # Remove day of week if present
        date_parts = date_string.split()
        
        # If first part looks like a day of week, remove it
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if len(date_parts) > 0 and date_parts[0].lower() in days_of_week:
            date_parts = date_parts[1:]
        
        # Extract year if present
        year = None
        day = None
        month = None
        
        # Look for a 4-digit year in the date parts
        for i, part in enumerate(date_parts):
            if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2100:
                year = int(part)
                # Remove year from date_parts for further processing
                date_parts = date_parts[:i] + date_parts[i+1:]
                break
        
        # If no year found, use default_year or current year
        if year is None:
            if default_year:
                year = default_year
            else:
                year = datetime.now().year
        
        if len(date_parts) >= 2:
            day = date_parts[0]
            month = date_parts[1]
            
            # Month name to number mapping
            month_map = {
                'january': 1, 'jan': 1,
                'february': 2, 'feb': 2,
                'march': 3, 'mar': 3,
                'april': 4, 'apr': 4,
                'may': 5,
                'june': 6, 'jun': 6,
                'july': 7, 'jul': 7,
                'august': 8, 'aug': 8,
                'september': 9, 'sep': 9, 'sept': 9,
                'october': 10, 'oct': 10,
                'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }
            
            month_num = month_map.get(month.lower())
            if month_num and day.isdigit():
                formatted_date = f"{year}-{month_num:02d}-{int(day):02d}"
                return formatted_date
    
    except Exception as e:
        print(f"Error parsing date '{date_string}': {e}")
    
    # If parsing fails, return the original string
    return date_string

def scrape_league_rounds(base_url, max_rounds=30, season_year=None):
    """
    Scrape match URLs and dates from all FotMob league rounds
    """
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    all_matches = []
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Navigate through each round
        for round_num in range(max_rounds):
            print(f"Scraping round {round_num + 1}...")
            
            # Construct URL for current round
            round_url = base_url.replace('round=0', f'round={round_num}')
            
            try:
                driver.get(round_url)
                time.sleep(3)
                
                # Check if round exists
                match_elements = driver.find_elements(By.CSS_SELECTOR, 'a.css-1ajdexg-MatchWrapper')
                if not match_elements:
                    print(f"No matches found in round {round_num + 1}, stopping...")
                    break
                
                # Extract match information with proper date handling
                round_matches = 0
                current_date = None
                
                for match_element in match_elements:
                    try:
                        # Find the date header that precedes this match
                        match_date = None
                        try:
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", match_element)
                            
                            preceding_date = driver.execute_script("""
                                var match = arguments[0];
                                var current = match;
                                
                                while (current) {
                                    var prev = current.previousElementSibling;
                                    while (prev) {
                                        if (prev.tagName === 'H3' && prev.classList.contains('css-1fw9re8-HeaderCSS')) {
                                            return prev.textContent.trim();
                                        }
                                        prev = prev.previousElementSibling;
                                    }
                                    current = current.parentElement;
                                    if (current && current.tagName === 'BODY') break;
                                }
                                return null;
                            """, match_element)
                            
                            match_date = preceding_date if preceding_date else current_date
                            
                        except:
                            match_date = current_date
                        
                        if match_date != current_date:
                            current_date = match_date
                            print(f"  Date: {current_date}")
                        
                        # Convert date to ISO format
                        iso_date = parse_date_to_iso(match_date, season_year)
                        
                        # Get match URL
                        match_href = match_element.get_attribute('href')
                        if match_href:
                            if match_href.startswith('/'):
                                match_url = f"https://www.fotmob.com{match_href}"
                            else:
                                match_url = match_href
                            
                            # Extract team names
                            team_elements = match_element.find_elements(By.CSS_SELECTOR, '.css-1o142s8-TeamName')
                            if len(team_elements) >= 2:
                                home_team = team_elements[0].text.strip()
                                away_team = team_elements[1].text.strip()
                            else:
                                home_team = "Unknown"
                                away_team = "Unknown"
                            
                            # Extract score and status
                            score = "N/A"
                            try:
                                score_element = match_element.find_element(By.CSS_SELECTOR, '.css-1wwsq70-LSMatchStatusScore')
                                score = score_element.text.strip()
                            except:
                                pass
                            
                            status = "Unknown"
                            try:
                                status_element = match_element.find_element(By.CSS_SELECTOR, '.css-1t50dhw-StatusDotCSS')
                                status = status_element.get_attribute('title') or status_element.text.strip()
                            except:
                                try:
                                    status_element = match_element.find_element(By.CSS_SELECTOR, '.css-1ubkvjq-LSMatchStatusReason')
                                    status = status_element.text.strip()
                                except:
                                    pass
                            
                            match_data = {
                                'round': round_num + 1,
                                'date': iso_date,  # Now using ISO format
                                'home_team': home_team,
                                'away_team': away_team,
                                'score': score,
                                'status': status,
                                'match_url': match_url
                            }
                            
                            all_matches.append(match_data)
                            round_matches += 1
                            print(f"    {home_team} vs {away_team} - {score} [{status}] ({iso_date})")
                    
                    except Exception as e:
                        print(f"    Error processing match: {e}")
                        continue
                
                print(f"  Found {round_matches} matches in round {round_num + 1}")
                time.sleep(1)
                
            except Exception as e:
                print(f"Error accessing round {round_num + 1}: {e}")
                continue
        
        driver.quit()
        return all_matches
        
    except Exception as e:
        print(f"Error: {e}")
        if 'driver' in locals():
            driver.quit()
        return []

def scrape_match_xg(match_url, match_date, home_team, away_team, round_num):
    """
    Scrape xG values from a specific FotMob match page
    """
    
    # Ensure we're on the stats tab
    if ':tab=stats' not in match_url:
        # Handle URLs that already have fragments vs those that don't
        if '#' in match_url:
            # URL like: https://www.fotmob.com/matches/team1-vs-team2/abc#4757737
            # Should become: https://www.fotmob.com/matches/team1-vs-team2/abc#4757737:tab=stats
            match_url = match_url + ':tab=stats'
        else:
            # URL like: https://www.fotmob.com/matches/team1-vs-team2/abc
            # Should become: https://www.fotmob.com/matches/team1-vs-team2/abc#:tab=stats
            match_url = match_url + '#:tab=stats'
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.get(match_url)
        time.sleep(5)
        
        # Get team ID mapping from logos
        team_id_mapping = {}
        try:
            all_logos = driver.find_elements(By.CSS_SELECTOR, 'img[src*="teamlogo"]')
            header_logos = [logo for logo in all_logos if '_small.png' in logo.get_attribute('src') or '_medium.png' in logo.get_attribute('src')]
            
            processed_ids = set()
            for logo in header_logos:
                logo_url = logo.get_attribute('src')
                team_id_match = re.search(r'teamlogo/(\d+)_', logo_url)
                if team_id_match:
                    team_id = team_id_match.group(1)
                    if team_id in processed_ids:
                        continue
                    processed_ids.add(team_id)
                    
                    try:
                        parent = logo.find_element(By.XPATH, '..')
                        team_name_text = parent.text.strip()
                        if not team_name_text or len(team_name_text) < 3:
                            grandparent = parent.find_element(By.XPATH, '..')
                            team_name_text = grandparent.text.strip()
                        
                        if team_name_text:
                            team_name_clean = re.sub(r'\d+', '', team_name_text)
                            team_name_clean = re.sub(r'[\'"]', '', team_name_clean)
                            team_name_clean = re.sub(r'\s+', ' ', team_name_clean).strip()
                            
                            if len(team_name_clean) > 2:
                                team_id_mapping[team_id] = team_name_clean
                    except:
                        pass
        except:
            pass
        
        all_shots = []
        seen_shots = set()
        shot_count = 0
        max_attempts = 50
        
        # Navigate through shots
        while shot_count < max_attempts:
            current_shot = {}
            
            # Get xG values
            xg_elements = driver.find_elements(By.CSS_SELECTOR, '.css-g9mdo5-XGItemValue span')
            if xg_elements:
                xg_values = []
                for element in xg_elements:
                    text = element.text.strip()
                    if text and '.' in text:
                        try:
                            xg_values.append(float(text))
                        except ValueError:
                            continue
                
                if len(xg_values) >= 2:
                    current_shot['xg'] = xg_values[0]
                    current_shot['xgot'] = xg_values[1]
                elif len(xg_values) == 1:
                    current_shot['xg'] = xg_values[0]
                    current_shot['xgot'] = None
            
            # Get shot result
            try:
                shot_info_container = driver.find_element(By.CSS_SELECTOR, '.css-1jl5w6u-ShotInfo')
                spans = shot_info_container.find_elements(By.CSS_SELECTOR, 'span')
                
                result_value = 'Unknown'
                for i, span in enumerate(spans):
                    if span.text.strip() == 'Result' and i + 1 < len(spans):
                        result_value = spans[i + 1].text.strip()
                        break
                
                current_shot['result'] = result_value
            except:
                current_shot['result'] = 'Unknown'
            
            # Get player info and extract minute
            try:
                player_container = driver.find_element(By.CSS_SELECTOR, '.css-108cnc0-PlayerNameContainer')
                player_text = player_container.text.strip()
                
                # Extract minute
                minute_match = re.search(r'(\d+)(?:\s*\+\s*(\d+))?\'', player_text)
                if minute_match:
                    base_minute = int(minute_match.group(1))
                    added_time = int(minute_match.group(2)) if minute_match.group(2) else 0
                    current_shot['minute'] = base_minute + added_time
                else:
                    current_shot['minute'] = None
                
                # Extract player name
                parts = player_text.split('\n')
                if len(parts) > 1:
                    player_name = parts[-1].strip()
                else:
                    player_name = re.sub(r'^\d+\'?\s*\+?\s*\d*\'?\s*', '', player_text).strip()
                
                current_shot['player_name'] = player_name
                
                # Get team info
                try:
                    team_img = player_container.find_element(By.CSS_SELECTOR, 'img.TeamIcon')
                    team_logo_url = team_img.get_attribute('src')
                    team_id_match = re.search(r'teamlogo/(\d+)_', team_logo_url)
                    if team_id_match:
                        current_shot['team_id'] = team_id_match.group(1)
                except:
                    current_shot['team_id'] = 'unknown'
                    
            except:
                current_shot['player_name'] = f"Player {shot_count + 1}"
                current_shot['team_id'] = 'unknown'
                current_shot['minute'] = None
            
            # Check for duplicates
            shot_identifier = f"{current_shot.get('player_name', '')}-{current_shot.get('xg', 0)}-{current_shot.get('minute', 0)}"
            if shot_identifier in seen_shots:
                break
            seen_shots.add(shot_identifier)
            
            # Add shot if it has xG data
            if current_shot.get('xg') is not None:
                all_shots.append(current_shot)
            
            # Navigate to next shot
            try:
                next_buttons = driver.find_elements(By.CSS_SELECTOR, '.css-65dgen-DropdownBrowseButton')
                if len(next_buttons) >= 2:
                    next_button = next_buttons[1]
                    if next_button.is_enabled() and next_button.is_displayed():
                        driver.execute_script("arguments[0].click();", next_button)
                        time.sleep(1.5)
                        shot_count += 1
                    else:
                        break
                else:
                    break
            except:
                break
        
        driver.quit()
        
        # Map team IDs to names
        team_ids = set(shot.get('team_id') for shot in all_shots if shot.get('team_id') != 'unknown')
        team_id_to_name = {}
        
        if team_id_mapping:
            team_id_to_name = team_id_mapping
        elif len(team_ids) == 2:
            team_list = sorted(list(team_ids))
            team_id_to_name[team_list[0]] = home_team
            team_id_to_name[team_list[1]] = away_team
        else:
            for team_id in team_ids:
                team_id_to_name[team_id] = f'Team {team_id}'
        
        # Prepare shot data with match info
        shot_details = []
        for shot in all_shots:
            team_id = shot.get('team_id', 'unknown')
            team_name = team_id_to_name.get(team_id, f'Team {team_id}')
            
            shot_details.append({
                'round': round_num,
                'match_date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'match_url': match_url,
                'player_name': shot['player_name'],
                'team_name': team_name,
                'minute': shot.get('minute'),
                'xg': shot.get('xg'),
                'xgot': shot.get('xgot'),
                'result': shot.get('result')
            })
        
        return shot_details
        
    except Exception as e:
        print(f"Error scraping xG for {home_team} vs {away_team}: {e}")
        if 'driver' in locals():
            driver.quit()
        return []

def scrape_full_league_xg(base_url, max_rounds=30, start_from_round=1, season_year=None):
    """
    Scrape all matches and their xG data from a league
    """
    
    print("Step 1: Getting all match URLs...")
    matches = scrape_league_rounds(base_url, max_rounds, season_year)
    
    if not matches:
        print("No matches found!")
        return [], []
    
    # Save the matches summary CSV first
    save_matches_summary_to_csv(matches, 'league_matches_summary.csv')
    
    # Filter matches if starting from specific round
    if start_from_round > 1:
        matches = [m for m in matches if m['round'] >= start_from_round]
        print(f"Filtered to {len(matches)} matches from round {start_from_round} onwards")
    
    print(f"\nStep 2: Scraping xG data for {len(matches)} matches...")
    
    all_shots = []
    completed_matches = 0
    
    for i, match in enumerate(matches, 1):
        print(f"\nProcessing match {i}/{len(matches)}: {match['home_team']} vs {match['away_team']} (Round {match['round']})")
        
        # Only scrape matches that have finished (have scores)
        if match['status'] in ['Full time', 'FT'] and match['score'] != 'N/A':
            shots = scrape_match_xg(
                match['match_url'], 
                match['date'], 
                match['home_team'], 
                match['away_team'],
                match['round']
            )
            
            if shots:
                all_shots.extend(shots)
                print(f"  Found {len(shots)} shots")
                completed_matches += 1
            else:
                print("  No shot data found")
        else:
            print(f"  Skipping - Match not finished or no score available (Status: {match['status']})")
        
        # Small delay between matches
        time.sleep(2)
    
    print(f"\nCompleted! Processed {completed_matches} matches with {len(all_shots)} total shots")
    return all_shots, matches

def save_shots_to_csv(shots, filename='league_xg_data.csv'):
    """Save all shot data to CSV file"""
    if not shots:
        print("No shot data to save")
        return
    
    # Debug: Check what we're actually getting
    print(f"Debug: shots type = {type(shots)}")
    if shots:
        print(f"Debug: first item type = {type(shots[0])}")
        print(f"Debug: first item = {shots[0]}")
    
    # Ensure we have a list of dictionaries
    if not isinstance(shots, list):
        print("Error: shots is not a list")
        return
    
    if not shots or not isinstance(shots[0], dict):
        print("Error: shots does not contain dictionaries")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['round', 'match_date', 'home_team', 'away_team', 'match_url', 
                     'player_name', 'team_name', 'minute', 'xg', 'xgot', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for shot in shots:
            if isinstance(shot, dict):
                writer.writerow(shot)
            else:
                print(f"Warning: Skipping non-dict item: {shot}")
    
    print(f"Shot data saved to {filename}")

def save_matches_summary_to_csv(matches, filename='league_matches_summary.csv'):
    """Save match summary data to CSV file"""
    if not matches:
        print("No match data to save")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['round', 'match_date', 'home_team', 'home_goals', 'away_team', 'away_goals', 'match_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for match in matches:
            # Parse the score to extract home and away goals
            home_goals = None
            away_goals = None
            
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
    
    print(f"Match summary saved to {filename}")

# Usage
if __name__ == "__main__":
    base_url = "https://www.fotmob.com/en-GB/leagues/47/matches/premier-league?season=2024-2025&group=by-round&round=0"
    
    # Scrape all matches and their xG data
    # You can specify season_year if you know it, otherwise it will use current year for dates without year
    result = scrape_full_league_xg(base_url, max_rounds=38, start_from_round=1, season_year=2024)
    
    # Handle the return value properly
    if isinstance(result, tuple) and len(result) == 2:
        all_shots, matches = result
    else:
        all_shots = result
        matches = []
    
    if all_shots:
        save_shots_to_csv(all_shots, 'prem_complete_xg_data.csv')
        
        # Print summary
        matches_processed = len(set(shot['match_url'] for shot in all_shots if isinstance(shot, dict)))
        rounds_processed = len(set(shot['round'] for shot in all_shots if isinstance(shot, dict)))
        
        print(f"\n=== SUMMARY ===")
        print(f"Rounds processed: {rounds_processed}")
        print(f"Matches processed: {matches_processed}")
        print(f"Total shots: {len(all_shots)}")
        print(f"Data saved to: superligaen_complete_xg_data.csv")
    else:
        print("No data collected")