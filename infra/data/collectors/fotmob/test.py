from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import json
import time

chrome_options = Options()
chrome_options.add_argument("--headless")

driver = webdriver.Chrome(options=chrome_options)

# Navigate to the SAME URL you used in the browser console
# This is the Premier League overview page where the match data was found
driver.get('https://www.fotmob.com/en-GB/leagues/47/overview/premier-league?season=2024-2025')
time.sleep(8)

soup = BeautifulSoup(driver.page_source, 'html.parser')
script = soup.find('script', {'id': '__NEXT_DATA__'})

if script and script.string:
    try:
        full_data = json.loads(script.string)
        
        print("=== Now on the Premier League overview page ===")
        print(f"Current page: {full_data.get('page', 'Unknown')}")
        print(f"Current URL path: {full_data.get('props', {}).get('url', 'Unknown')}")
        
        page_props = full_data.get('props', {}).get('pageProps', {})
        print(f"Available keys in pageProps: {list(page_props.keys())}")
        
        # Now try the paths that worked in the browser console
        paths_to_check = [
            ['overview', 'leagueOverviewMatches'],
            ['matches', 'allMatches']
        ]
        
        for path_list in paths_to_check:
            try:
                current_data = page_props
                path_str = '.'.join(path_list)
                
                for key in path_list:
                    current_data = current_data[key]
                
                print(f"\n--- Checking path: pageProps.{path_str} ---")
                print(f"Found {len(current_data)} items")
                
                # Look for our specific match
                for i, match in enumerate(current_data):
                    match_id = match.get('id')
                    if str(match_id) == '4506263':
                        print(f"✓ FOUND MATCH 4506263 at index {i}!")
                        
                        print(f"Match details:")
                        print(f"  ID: {match.get('id')}")
                        print(f"  Home: {match.get('home', {}).get('name', 'N/A')}")
                        print(f"  Away: {match.get('away', {}).get('name', 'N/A')}")
                        print(f"  Date: {match.get('time', {}).get('utcTime', 'N/A')}")
                        print(f"  Finished: {match.get('status', {}).get('finished', 'N/A')}")
                        print(f"  Score: {match.get('status', {}).get('scoreStr', 'N/A')}")
                        
                        # Create DataFrame
                        df_match = pd.json_normalize(match)
                        shots_df = pd.json_normalize(df_match['content']['shotmap']['shots'])
                        print(f"\n*** SUCCESS! ***")
                        print(f"DataFrame shape: {df_match.shape}")
                        print(f"DataFrame columns: {list(df_match.columns)}")
                        
                        # Display the DataFrame
                        print(f"\nMatch data:")
                        print(df_match.to_string())

                        print(shots_df)
                        
                        break
                else:
                    print(f"  Match 4506263 not found in {path_str}")
                    # Show what match IDs are available
                    available_ids = [str(m.get('id', 'No ID')) for m in current_data[:5]]
                    print(f"  First 5 available IDs: {available_ids}")
                        
            except (KeyError, TypeError) as e:
                print(f"Error accessing path pageProps.{'.'.join(path_list)}: {e}")
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")

driver.quit()