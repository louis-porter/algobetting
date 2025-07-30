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

def scrape_league_rounds(base_url, max_rounds=30):
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
                
                # Check if round exists (no matches found indicates end of rounds)
                match_elements = driver.find_elements(By.CSS_SELECTOR, 'a.css-1ajdexg-MatchWrapper')
                if not match_elements:
                    print(f"No matches found in round {round_num + 1}, stopping...")
                    break
                
                # Get all date headers and matches, grouping them properly
                date_headers = driver.find_elements(By.CSS_SELECTOR, 'h3.css-1fw9re8-HeaderCSS')
                
                # Extract match information
                round_matches = 0
                current_date = None
                
                # Process each match and find its corresponding date
                for match_element in match_elements:
                    try:
                        # Find the date header that precedes this match
                        match_date = None
                        try:
                            # Get the position of this match element
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", match_element)
                            
                            # Find the preceding date header by walking up the DOM
                            # Use JavaScript to find the closest preceding h3 date header
                            preceding_date = driver.execute_script("""
                                var match = arguments[0];
                                var current = match;
                                
                                // Walk backwards through siblings and parent siblings
                                while (current) {
                                    // Check previous siblings
                                    var prev = current.previousElementSibling;
                                    while (prev) {
                                        if (prev.tagName === 'H3' && prev.classList.contains('css-1fw9re8-HeaderCSS')) {
                                            return prev.textContent.trim();
                                        }
                                        prev = prev.previousElementSibling;
                                    }
                                    
                                    // Move up to parent and continue searching
                                    current = current.parentElement;
                                    if (current && current.tagName === 'BODY') break;
                                }
                                
                                return null;
                            """, match_element)
                            
                            match_date = preceding_date if preceding_date else current_date
                            
                        except:
                            match_date = current_date
                        
                        # Only print date change when it actually changes
                        if match_date != current_date:
                            current_date = match_date
                            print(f"  Date: {current_date}")
                        
                        # Get match URL
                        match_href = match_element.get_attribute('href')
                        if match_href:
                            # Convert relative URL to absolute
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
                            
                            # Extract score if available
                            score = "N/A"
                            try:
                                score_element = match_element.find_element(By.CSS_SELECTOR, '.css-1wwsq70-LSMatchStatusScore')
                                score = score_element.text.strip()
                            except:
                                pass
                            
                            # Extract match status
                            status = "Unknown"
                            try:
                                status_element = match_element.find_element(By.CSS_SELECTOR, '.css-1t50dhw-StatusDotCSS')
                                status = status_element.get_attribute('title') or status_element.text.strip()
                            except:
                                # Try alternative status selector
                                try:
                                    status_element = match_element.find_element(By.CSS_SELECTOR, '.css-1ubkvjq-LSMatchStatusReason')
                                    status = status_element.text.strip()
                                except:
                                    pass
                            
                            match_data = {
                                'round': round_num + 1,
                                'date': match_date,
                                'home_team': home_team,
                                'away_team': away_team,
                                'score': score,
                                'status': status,
                                'match_url': match_url
                            }
                            
                            all_matches.append(match_data)
                            round_matches += 1
                            print(f"    {home_team} vs {away_team} - {score} [{status}]")
                    
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

def save_matches_to_csv(matches, filename='league_matches.csv'):
    """Save match data to CSV file"""
    if not matches:
        print("No match data to save")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['round', 'date', 'home_team', 'away_team', 'score', 'status', 'match_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for match in matches:
            writer.writerow(match)
    
    print(f"Match data saved to {filename}")

# Usage
if __name__ == "__main__":
    base_url = "https://www.fotmob.com/en-GB/leagues/46/matches/superligaen?season=2024-2025&group=by-round&round=0"
    
    print("Scraping all rounds...")
    matches = scrape_league_rounds(base_url, max_rounds=50)
    
    if matches:
        print(f"\nTotal matches found: {len(matches)}")
        save_matches_to_csv(matches, 'superligaen_matches.csv')
        
        # Print summary by round
        rounds_summary = {}
        for match in matches:
            round_num = match['round']
            if round_num not in rounds_summary:
                rounds_summary[round_num] = 0
            rounds_summary[round_num] += 1
        
        print("\nRounds summary:")
        for round_num in sorted(rounds_summary.keys()):
            print(f"  Round {round_num}: {rounds_summary[round_num]} matches")
    else:
        print("No matches found")