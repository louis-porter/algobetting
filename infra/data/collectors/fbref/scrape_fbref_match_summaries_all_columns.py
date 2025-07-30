import requests
from bs4 import BeautifulSoup
import time
import json

def scrape_fotmob_xg(match_url):
    """
    Simple scraper to get xG values from FotMob match page
    """
    
    # Headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Make the request
        response = requests.get(match_url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find xG values using the CSS class you identified
        xg_elements = soup.find_all('span', class_='css-g9mdo5-XGItemValue')
        
        # Extract xG values
        xg_values = []
        for element in xg_elements:
            # Get the nested span with the actual value
            value_span = element.find('span')
            if value_span:
                xg_values.append(float(value_span.text.strip()))
        
        # Also try to get player info from the same containers
        shot_info = []
        browse_containers = soup.find_all('div', class_='css-gylia7-BrowseContainer')
        
        for container in browse_containers:
            # Look for player name container
            player_container = container.find('div', class_='css-108cnc0-PlayerNameContainer')
            if player_container:
                # Extract time and player name
                text_content = player_container.get_text(strip=True)
                
                # Find xG value in this container
                xg_element = container.find('span', class_='css-g9mdo5-XGItemValue')
                xg_value = None
                if xg_element:
                    xg_span = xg_element.find('span')
                    if xg_span:
                        xg_value = float(xg_span.text.strip())
                
                shot_info.append({
                    'player_info': text_content,
                    'xg_value': xg_value
                })
        
        return {
            'xg_values': xg_values,
            'shot_details': shot_info,
            'total_shots': len(xg_values)
        }
        
    except requests.RequestException as e:
        print(f"Error fetching the page: {e}")
        return None
    except Exception as e:
        print(f"Error parsing the page: {e}")
        return None

def scrape_with_selenium(match_url):
    """
    Alternative approach using Selenium for JavaScript-heavy content
    """
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in background
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        # Initialize driver with auto-downloaded ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print("Loading page...")
        driver.get(match_url)
        
        # Wait longer for content to load
        time.sleep(5)
        
        print("Looking for shot carousel...")
        
        # First, let's try to get team names and their IDs from the page header
        team_names = {}
        team_id_mapping = {}
        
        try:
            # Look for team logos specifically in the match header area (not in shot details)
            header_logos = driver.find_elements(By.CSS_SELECTOR, 'header img[src*="teamlogo"], [class*="header"] img[src*="teamlogo"], [class*="match"] img[src*="teamlogo"]')
            
            # If no header logos found, look for larger logos (small vs xsmall)
            if not header_logos:
                all_logos = driver.find_elements(By.CSS_SELECTOR, 'img[src*="teamlogo"]')
                # Filter for larger logos (small, medium) and avoid the xsmall ones used in shot details
                header_logos = [logo for logo in all_logos if '_small.png' in logo.get_attribute('src') or '_medium.png' in logo.get_attribute('src')]
            
            print("Found team logos in header:")
            processed_ids = set()
            
            for logo in header_logos:
                logo_url = logo.get_attribute('src')
                # Extract team ID from logo URL
                import re
                team_id_match = re.search(r'teamlogo/(\d+)_', logo_url)
                if team_id_match:
                    team_id = team_id_match.group(1)
                    
                    # Skip if we already processed this team ID
                    if team_id in processed_ids:
                        continue
                    processed_ids.add(team_id)
                    
                    print(f"  Team ID {team_id}: {logo_url}")
                    
                    # Try to find the team name near this logo
                    try:
                        # Look for text elements near the logo
                        parent = logo.find_element(By.XPATH, '..')
                        team_name_text = parent.text.strip()
                        
                        # If parent doesn't have good text, try grandparent
                        if not team_name_text or len(team_name_text) < 3:
                            grandparent = parent.find_element(By.XPATH, '..')
                            team_name_text = grandparent.text.strip()
                        
                        # Clean up the team name
                        if team_name_text:
                            # Remove numbers, extra whitespace, and common unwanted text
                            team_name_clean = re.sub(r'\d+', '', team_name_text)  # Remove numbers
                            team_name_clean = re.sub(r'[\'"]', '', team_name_clean)  # Remove quotes
                            team_name_clean = re.sub(r'\s+', ' ', team_name_clean).strip()  # Clean whitespace
                            
                            # Skip if it's too short or looks like a player name
                            if len(team_name_clean) > 2 and not team_name_clean.lower().startswith('edward'):
                                team_id_mapping[team_id] = team_name_clean
                                print(f"    Mapped to: {team_name_clean}")
                    except:
                        pass
            
            # Extract team names from page title as fallback
            title_element = driver.find_element(By.TAG_NAME, 'title')
            title_text = title_element.get_attribute('innerHTML')
            
            vs_match = re.search(r'([^-]+?)\s+(?:vs?\.?|v\.?|-)\s+([^-]+?)(?:\s|$|\()', title_text, re.IGNORECASE)
            if vs_match:
                team1 = vs_match.group(1).strip()
                team2 = vs_match.group(2).strip()
                print(f"Found teams from title: {team1} vs {team2}")
                team_names['team1'] = team1
                team_names['team2'] = team2
                            
        except Exception as e:
            print(f"Could not extract team info from page: {e}")
            team_names = {}
            team_id_mapping = {}
        
        all_shots = []
        seen_shots = set()  # Track shots we've already seen
        shot_count = 0
        max_attempts = 50  # Safety limit
        
        # Look for the right arrow button to navigate through shots
        try:
            while shot_count < max_attempts:
                current_shot = {}
                
                # Get xG values for current shot
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
                    
                    # Separate xG and xGOT
                    if len(xg_values) >= 2:
                        current_shot['xg'] = xg_values[0]  # First value is xG
                        current_shot['xgot'] = xg_values[1]  # Second value is xGOT
                    elif len(xg_values) == 1:
                        current_shot['xg'] = xg_values[0]
                        current_shot['xgot'] = None
                    
                    current_shot['xg_values'] = xg_values  # Keep original for compatibility
                
                # Get shot result (Goal, Save, Miss, etc.)
                try:
                    # Look for result text/indicators in the shot carousel
                    result_elements = driver.find_elements(By.CSS_SELECTOR, '[class*="result"], [class*="Result"], [class*="outcome"], [class*="Outcome"]')
                    
                    shot_result = None
                    for result_element in result_elements:
                        result_text = result_element.text.strip()
                        if result_text and any(keyword in result_text.lower() for keyword in ['goal', 'save', 'miss', 'block', 'post', 'bar', 'wide', 'over']):
                            shot_result = result_text
                            break
                    
                    # If no specific result element found, try to infer from other indicators
                    if not shot_result:
                        # Check if there are any goal indicators or other result clues
                        all_text_elements = driver.find_elements(By.CSS_SELECTOR, 'span, div')
                        for element in all_text_elements:
                            text = element.text.strip().lower()
                            if text in ['goal', 'save', 'saved', 'miss', 'missed', 'blocked', 'post', 'crossbar', 'wide', 'over']:
                                shot_result = text.title()
                                break
                    
                    current_shot['result'] = shot_result or 'Unknown'
                    
                except Exception as e:
                    current_shot['result'] = 'Unknown'
                
                # Get player info and team ID for current shot
                try:
                    player_container = driver.find_element(By.CSS_SELECTOR, '.css-108cnc0-PlayerNameContainer')
                    player_text = player_container.text.strip()
                    current_shot['player_info'] = player_text
                    
                    # Try to get team info from the team icon
                    try:
                        team_img = player_container.find_element(By.CSS_SELECTOR, 'img.TeamIcon')
                        team_logo_url = team_img.get_attribute('src')
                        
                        # Extract team ID from logo URL (e.g., teamlogo/8113_xsmall.png -> 8113)
                        import re
                        team_id_match = re.search(r'teamlogo/(\d+)_', team_logo_url)
                        if team_id_match:
                            current_shot['team_id'] = team_id_match.group(1)
                        
                        current_shot['team_logo_url'] = team_logo_url
                        
                    except:
                        current_shot['team_id'] = 'unknown'
                        
                except:
                    current_shot['player_info'] = f"Shot {shot_count + 1}"
                    current_shot['team_id'] = 'unknown'
                
                # Create a unique identifier for this shot
                shot_identifier = f"{current_shot.get('player_info', '')}-{current_shot.get('xg_values', [])}"
                
                # Check if we've seen this shot before (indicates we've looped back)
                if shot_identifier in seen_shots:
                    print("Detected loop - we've seen this shot before. Stopping.")
                    break
                
                # Add to seen shots and our results
                seen_shots.add(shot_identifier)
                
                # Add current shot to list if it has data
                if current_shot.get('xg_values'):
                    all_shots.append(current_shot)
                    
                    # Format the display
                    team_id = current_shot.get('team_id', 'unknown')
                    xg_display = f"xG: {current_shot.get('xg', 'N/A')}"
                    if current_shot.get('xgot') is not None:
                        xg_display += f", xGOT: {current_shot.get('xgot')}"
                    
                    result_display = f" [{current_shot.get('result', 'Unknown')}]" if current_shot.get('result') != 'Unknown' else ""
                    
                    print(f"Shot {len(all_shots)}: {current_shot['player_info']} [Team {team_id}] - {xg_display}{result_display}")
                
                # Try to find and click the next button
                try:
                    next_buttons = driver.find_elements(By.CSS_SELECTOR, '.css-65dgen-DropdownBrowseButton')
                    
                    if len(next_buttons) >= 2:
                        next_button = next_buttons[1]  # Right arrow
                        
                        if next_button.is_enabled() and next_button.is_displayed():
                            driver.execute_script("arguments[0].click();", next_button)
                            time.sleep(1.5)  # Wait for content to load
                            shot_count += 1
                        else:
                            print("Next button disabled - reached end of shots")
                            break
                    else:
                        print("Navigation buttons not found")
                        break
                        
                except Exception as e:
                    print(f"Error navigating to next shot: {e}")
                    break
            
            # Now analyze team IDs and map to actual team names
            print(f"\nAnalyzing {len(all_shots)} unique shots...")
            
            # Get unique team IDs
            team_ids = set()
            for shot in all_shots:
                if shot.get('team_id') and shot['team_id'] != 'unknown':
                    team_ids.add(shot['team_id'])
            
            print(f"Found team IDs: {list(team_ids)}")
            
            # Try to map team IDs to team names using the logo mapping we found
            team_id_to_name = {}
            
            if team_id_mapping:
                # We have direct ID-to-name mapping from logos - use this!
                print(f"Using logo-based mapping: {team_id_mapping}")
                team_id_to_name = team_id_mapping
            elif len(team_ids) == 2 and team_names.get('team1') and team_names.get('team2'):
                # Fallback to title-based mapping
                team_list = list(team_ids)
                sorted_ids = sorted(team_list)
                
                team_id_to_name[sorted_ids[0]] = team_names['team1']
                team_id_to_name[sorted_ids[1]] = team_names['team2']
                
                print(f"Using title-based mapping:")
                print(f"  Team {sorted_ids[0]} -> {team_names['team1']}")
                print(f"  Team {sorted_ids[1]} -> {team_names['team2']}")
            else:
                # Last resort - generic names
                for team_id in team_ids:
                    team_id_to_name[team_id] = f'Team {team_id}'
                print(f"Using generic mapping: {team_id_to_name}")
            
            # Compile results with team name mapping
            xg_values = []
            shot_details = []
            team_stats = {}
            
            for shot in all_shots:
                team_id = shot.get('team_id', 'unknown')
                team_name = team_id_to_name.get(team_id, f'Team {team_id}')
                
                # Initialize team stats
                if team_name not in team_stats:
                    team_stats[team_name] = {
                        'shots': 0,
                        'total_xg': 0.0,
                        'total_xgot': 0.0,
                        'team_id': team_id
                    }
                
                # Add to team stats
                team_stats[team_name]['shots'] += 1
                team_stats[team_name]['total_xg'] += shot.get('xg', 0)
                if shot.get('xgot'):
                    team_stats[team_name]['total_xgot'] += shot.get('xgot', 0)
                
                shot_details.append({
                    'player_info': shot['player_info'],
                    'team_id': team_id,
                    'team_name': team_name,
                    'xg': shot.get('xg'),
                    'xgot': shot.get('xgot'),
                    'result': shot.get('result'),
                    'xg_values': shot['xg_values']
                })
                xg_values.extend(shot['xg_values'])
            
            # Print summary without extra verification
            print(f"\n=== DETAILED SUMMARY ===")
            for team_name, stats in team_stats.items():
                print(f"{team_name} (ID: {stats['team_id']}):")
                print(f"  Shots: {stats['shots']}")
                print(f"  Total xG: {stats['total_xg']:.2f}")
                if stats['total_xgot'] > 0:
                    print(f"  Total xGOT: {stats['total_xgot']:.2f}")
                print()
            
        except Exception as e:
            print(f"Error in shot navigation: {e}")
            # Fallback to original method
            xg_elements = driver.find_elements(By.CSS_SELECTOR, '.css-g9mdo5-XGItemValue span')
            xg_values = []
            shot_details = []
            
            for element in xg_elements:
                text = element.text.strip()
                if text and '.' in text:
                    try:
                        xg_values.append(float(text))
                    except ValueError:
                        continue
        
        # If still no luck, dump the page source to see what's there
        if not xg_values:
            print("No xG values found. Checking page source...")
            page_source = driver.page_source
            
            # Look for any decimal numbers that might be xG values
            import re
            decimal_pattern = r'\b0\.\d{2,3}\b'
            potential_xg = re.findall(decimal_pattern, page_source)
            
            if potential_xg:
                print(f"Found potential xG values in source: {potential_xg}")
                xg_values = [float(x) for x in potential_xg]
            else:
                # Save page source for debugging
                with open('fotmob_debug.html', 'w', encoding='utf-8') as f:
                    f.write(page_source)
                print("Saved page source to fotmob_debug.html for inspection")
        
        driver.quit()
        
        return {
            'xg_values': xg_values,
            'shot_details': shot_details,
            'total_shots': len(xg_values)
        }
        
    except Exception as e:
        print(f"Selenium error: {e}")
        if 'driver' in locals():
            driver.quit()
        return None

# Usage example
if __name__ == "__main__":
    match_url = "https://www.fotmob.com/en-GB/matches/fc-midtjylland-vs-randers-fc/29a4m4#4757639:tab=stats"
    
    print("Trying Selenium approach...")
    result = scrape_with_selenium(match_url)
    
    if result and result['xg_values']:
        print(f"Found {result['total_shots']} shots")
        print("xG Values:", result['xg_values'])
        print("\nShot Details:")
        for shot in result['shot_details']:
            print(f"  {shot['player_info']}: {shot['xg_value']}")
    else:
        print("Selenium approach also failed. The match might not have detailed xG data yet.")