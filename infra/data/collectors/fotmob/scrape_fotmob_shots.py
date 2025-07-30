import requests
from bs4 import BeautifulSoup
import time
import json
import csv
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

def scrape_fotmob_xg(match_url):
    """
    Scrape xG values from FotMob match page using Selenium
    """
    
    # Setup Chrome options
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
        
        print("Loading page...")
        driver.get(match_url)
        time.sleep(5)
        
        # Extract team names from page title
        team_names = {}
        try:
            title_element = driver.find_element(By.TAG_NAME, 'title')
            title_text = title_element.get_attribute('innerHTML')
            vs_match = re.search(r'([^-]+?)\s+(?:vs?\.?|v\.?|-)\s+([^-]+?)(?:\s|$|\()', title_text, re.IGNORECASE)
            if vs_match:
                team_names['team1'] = vs_match.group(1).strip()
                team_names['team2'] = vs_match.group(2).strip()
        except:
            team_names = {}
        
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
                # Find the "Result" title span and get the following value span
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
                
                # Extract minute from the text (e.g., "90'" or "90 + 3'")
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
                print(f"Shot {len(all_shots)}: {current_shot['player_name']} ({current_shot.get('minute', '?')}') - xG: {current_shot.get('xg', 'N/A')}")
            
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
        elif len(team_ids) == 2 and team_names.get('team1') and team_names.get('team2'):
            team_list = sorted(list(team_ids))
            team_id_to_name[team_list[0]] = team_names['team1']
            team_id_to_name[team_list[1]] = team_names['team2']
        else:
            for team_id in team_ids:
                team_id_to_name[team_id] = f'Team {team_id}'
        
        # Prepare final data
        shot_details = []
        for shot in all_shots:
            team_id = shot.get('team_id', 'unknown')
            team_name = team_id_to_name.get(team_id, f'Team {team_id}')
            
            shot_details.append({
                'player_name': shot['player_name'],
                'team_name': team_name,
                'minute': shot.get('minute'),
                'xg': shot.get('xg'),
                'xgot': shot.get('xgot'),
                'result': shot.get('result'),
                'match_url': match_url
            })
        
        return {
            'shot_details': shot_details,
            'total_shots': len(shot_details)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        if 'driver' in locals():
            driver.quit()
        return None

def save_to_csv(data, filename='fotmob_xg_data.csv'):
    """Save shot data to CSV file"""
    if not data or not data['shot_details']:
        print("No data to save")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['player_name', 'team_name', 'minute', 'xg', 'xgot', 'result', 'match_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for shot in data['shot_details']:
            writer.writerow(shot)
    
    print(f"Data saved to {filename}")

# Usage
if __name__ == "__main__":
    match_url = "https://www.fotmob.com/en-GB/matches/tottenham-hotspur-vs-arsenal/2sx06r#4506513:tab=stats"
    
    result = scrape_fotmob_xg(match_url)
    
    if result and result['shot_details']:
        print(f"\nFound {result['total_shots']} shots")
        save_to_csv(result, 'match_xg_data.csv')
    else:
        print("No shot data found")