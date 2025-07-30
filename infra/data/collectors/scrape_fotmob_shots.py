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
        
        # First, let's try to get team names from the page
        team_names = {}
        try:
            # Look for team names in various places on the page
            team_elements = driver.find_elements(By.CSS_SELECTOR, '[class*="team"], [class*="Team"], h1, h2, .match-header')
            
            for element in team_elements:
                text = element.text.strip()
                if 'vs' in text.lower() or 'v' in text:
                    # Try to extract team names from "Team A vs Team B" format
                    teams = text.replace(' vs ', '|').replace(' v ', '|').split('|')
                    if len(teams) == 2:
                        print(f"Found teams in text: {teams[0].strip()} vs {teams[1].strip()}")
            
            # Also get from URL
            if "midtjylland" in match_url.lower() and "randers" in match_url.lower():
                team_names['home'] = "FC Midtjylland"
                team_names['away'] = "Randers FC"
                print(f"Teams from URL: {team_names['home']} (home) vs {team_names['away']} (away)")
        except Exception as e:
            print(f"Error finding team names: {e}")
            team_names = {'home': 'Home Team', 'away': 'Away Team'}
        
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
                    
                    print(f"Shot {len(all_shots)}: {current_shot['player_info']} [Team {team_id}] - {xg_display}")
                
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
            
            # Try to map team IDs to team names
            # We can make educated guesses or look for more info on the page
            team_id_to_name = {}
            
            if len(team_ids) == 2:
                team_list = list(team_ids)
                # Since FC Midtjylland is home, try to determine which ID belongs to which team
                # This might require manual mapping or additional page analysis
                team_id_to_name[team_list[0]] = team_names.get('home', f'Team {team_list[0]}')
                team_id_to_name[team_list[1]] = team_names.get('away', f'Team {team_list[1]}')
            elif len(team_ids) == 1:
                # Only one team ID found - this might be an issue
                team_id = list(team_ids)[0]
                team_id_to_name[team_id] = f'Team {team_id}'
            
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
                        'total_xgot': 0.0
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
                    'xg_values': shot['xg_values']
                })
                xg_values.extend(shot['xg_values'])
            
            # Print detailed summary
            print(f"\n=== DETAILED SUMMARY ===")
            for team_name, stats in team_stats.items():
                print(f"{team_name}:")
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
        