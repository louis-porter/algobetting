from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import requests
from datetime import datetime

def create_driver():
    """Create and configure Chrome driver"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def get_all_season_matches(league_id=47, season="2024-2025"):
    """
    Get all matches from a season using the league overview page.
    This uses our working method that successfully found match 4506263.
    """
    driver = create_driver()
    
    try:
        # Navigate to the league overview page where all historical matches are stored
        url = f'https://www.fotmob.com/en-GB/leagues/{league_id}/overview/premier-league?season={season}'
        print(f"Getting all matches from: {url}")
        driver.get(url)
        time.sleep(8)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        script = soup.find('script', {'id': '__NEXT_DATA__'})

        if not script or not script.string:
            print("No data script found")
            return []

        # Parse the JSON data using our proven method
        full_data = json.loads(script.string)
        page_props = full_data.get('props', {}).get('pageProps', {})
        
        all_matches = []
        
        # Check both possible locations for match data (our working approach)
        search_paths = [
            ['matches', 'allMatches'],
            ['overview', 'leagueOverviewMatches']
        ]
        
        for path_list in search_paths:
            try:
                current_data = page_props
                for key in path_list:
                    current_data = current_data[key]
                
                print(f"Found {len(current_data)} matches in {'.'.join(path_list)}")
                
                # Extract all matches
                for match in current_data:
                    match_id = match.get('id')
                    if match_id:
                        # Get the pageUrl from the match data (this is the working format)
                        page_url = match.get('pageUrl', '')
                        
                        match_info = {
                            'match_id': match_id,
                            'home_team': match.get('home', {}).get('name', 'Unknown'),
                            'away_team': match.get('away', {}).get('name', 'Unknown'),
                            'date': match.get('status', {}).get('utcTime', ''),
                            'score': match.get('status', {}).get('scoreStr', ''),
                            'finished': match.get('status', {}).get('finished', False),
                            'round': match.get('round', 0),
                            'page_url': page_url  # Store the pageUrl for later use
                        }
                        all_matches.append(match_info)
                
                # If we found matches in the first path, we can break
                if all_matches:
                    break
                        
            except (KeyError, TypeError) as e:
                print(f"Path {'.'.join(path_list)} not found: {e}")
                continue
        
        print(f"Total matches found: {len(all_matches)}")
        return all_matches
        
    except Exception as e:
        print(f"Error getting season matches: {e}")
        return []
        
    finally:
        driver.quit()

def scrape_match_details(match_id, home_team_name, away_team_name, page_url=""):
    """
    Scrape detailed match data using multiple approaches.
    Combines our working match ID discovery with robust individual match access.
    """
    print(f"Scraping match {match_id}: {home_team_name} vs {away_team_name}")
    
    # Try multiple URL formats to find the working one
    url_attempts = []
    
    # If we have a pageUrl, try that first (from our working method)
    if page_url:
        url_attempts.append(f'https://www.fotmob.com{page_url}')
    
    # Add other potential URL formats
    url_attempts.extend([
        f'https://www.fotmob.com/matches/{match_id}',
        f'https://www.fotmob.com/match/{match_id}',
    ])
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.fotmob.com/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    # Try each URL format
    for i, match_url in enumerate(url_attempts):
        try:
            print(f"  Attempt {i+1}: {match_url}")
            response = requests.get(match_url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                print(f"    HTTP {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            script = soup.find('script', {'id': '__NEXT_DATA__'})
            
            if not script or not script.string:
                print(f"    No __NEXT_DATA__ found")
                continue
            
            # Check if this is a 404 page (like we discovered with invalid match IDs)
            if '"url":"/404"' in script.string:
                print(f"    Got 404 page")
                continue
            
            # Parse JSON data using our proven method
            strings = script.string
            before, sep, after = strings.partition(',"seo"')
            before = before.replace('{"props":{"pageProps":', '')
            before = before + '}'
            
            try:
                data = json.loads(before)
                print(f"  ✓ Successfully got data from {match_url}")
                return extract_shots_and_cards_from_data(data, match_id, home_team_name, away_team_name)
            except json.JSONDecodeError:
                print(f"    JSON parsing failed")
                continue
                
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    print(f"  ✗ All methods failed for match {match_id}")
    return None, None

def extract_shots_and_cards_from_data(data, match_id, home_team_name, away_team_name):
    """Extract shots and cards from parsed page data"""
    shots_data = []
    cards_data = []
    
    try:
        # Extract team ID mapping (same proven approach)
        team_id_to_name = {}
        general = data.get('general', {})
        if 'homeTeam' in general:
            home_team = general['homeTeam']
            if 'id' in home_team:
                team_id_to_name[str(home_team['id'])] = home_team.get('name', home_team_name)
        
        if 'awayTeam' in general:
            away_team = general['awayTeam']
            if 'id' in away_team:
                team_id_to_name[str(away_team['id'])] = away_team.get('name', away_team_name)
        
        # Extract shots data
        if 'content' in data and 'shotmap' in data['content'] and 'shots' in data['content']['shotmap']:
            shots = data['content']['shotmap']['shots']
            for shot in shots:
                team_id = str(shot.get('teamId', ''))
                team_name = team_id_to_name.get(team_id, 'Unknown')
                
                # Fallback team identification
                if team_name == 'Unknown':
                    # Simple alternating fallback (could be improved)
                    team_name = home_team_name if len([s for s in shots_data if s.get('team_name') == home_team_name]) <= len([s for s in shots_data if s.get('team_name') == away_team_name]) else away_team_name
                
                situation = shot.get('situation', '')
                event_type = 'penalty' if situation == 'Penalty' else 'shot'
                
                shots_data.append({
                    'match_id': match_id,
                    'team_name': team_name,
                    'player_name': shot.get('playerName', 'Unknown'),
                    'minute': shot.get('min'),
                    'event_type': event_type,
                    'situation': situation,
                    'expected_goals': shot.get('expectedGoals'),
                    'expected_goals_on_target': shot.get('expectedGoalsOnTarget'),
                    'x_coordinate': shot.get('x'),
                    'y_coordinate': shot.get('y'),
                    'blocked': shot.get('blocked', False),
                    'on_target': shot.get('onTarget', False)
                })
        
        # Extract cards data (both red and yellow)
        if 'content' in data and 'matchFacts' in data['content']:
            match_facts = data['content']['matchFacts']
            if 'events' in match_facts and 'events' in match_facts['events']:
                events = match_facts['events']['events']
                for event in events:
                    if event.get('card') in ['Red', 'Yellow']:
                        is_home = event.get('isHome', False)
                        team_name = home_team_name if is_home else away_team_name
                        
                        cards_data.append({
                            'match_id': match_id,
                            'team_name': team_name,
                            'player_name': event.get('fullName', 'Unknown'),
                            'minute': event.get('time'),
                            'card_type': event.get('card'),
                            'event_type': 'red_card' if event.get('card') == 'Red' else 'yellow_card',
                            'is_home': is_home
                        })
        
        print(f"    Found {len(shots_data)} shots and {len(cards_data)} cards")
        return shots_data, cards_data
        
    except Exception as e:
        print(f"    Error extracting data: {e}")
        return [], []

def scrape_season_data(league_id=47, season="2024-2025", max_matches=None):
    """
    Main function to scrape all shots and cards data for a season.
    Uses our proven approach: get match IDs from league overview, then scrape individual matches.
    """
    print(f"Starting to scrape {season} season data for league {league_id}")
    
    # Step 1: Get all matches from the season using our working method
    print("\nStep 1: Getting all matches from season...")
    matches = get_all_season_matches(league_id, season)
    
    if not matches:
        print("No matches found!")
        return
    
    # Filter to only finished matches
    finished_matches = [m for m in matches if m['finished']]
    print(f"Found {len(finished_matches)} finished matches out of {len(matches)} total matches")
    
    # Limit matches if specified (useful for testing)
    if max_matches:
        finished_matches = finished_matches[:max_matches]
        print(f"Limited to first {max_matches} matches for testing")
    
    # Save matches summary
    matches_df = pd.DataFrame(finished_matches)
    matches_df.to_csv(f'matches_{season.replace("-", "_")}.csv', index=False)
    print(f"Saved matches summary to matches_{season.replace('-', '_')}.csv")
    
    # Step 2: Scrape detailed data for each match
    print(f"\nStep 2: Scraping detailed data for {len(finished_matches)} matches...")
    
    all_shots = []
    all_cards = []
    success_count = 0
    
    for i, match in enumerate(finished_matches):
        print(f"\nProcessing {i+1}/{len(finished_matches)}: {match['home_team']} vs {match['away_team']}")
        
        shots_data, cards_data = scrape_match_details(
            match['match_id'],
            match['home_team'],
            match['away_team'],
            match.get('page_url', '')  # Pass the pageUrl if available
        )
        
        if shots_data is not None:
            success_count += 1
            
            # Add match context to shots data
            for shot in shots_data:
                shot.update({
                    'date': match['date'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'round': match['round']
                })
            all_shots.extend(shots_data)
            
            # Add match context to cards data
            if cards_data:
                for card in cards_data:
                    card.update({
                        'date': match['date'],
                        'home_team': match['home_team'],
                        'away_team': match['away_team'],
                        'round': match['round']
                    })
                all_cards.extend(cards_data)
        
        # Be respectful to the server
        time.sleep(2)
    
    # Step 3: Save all data to CSV files
    print(f"\nStep 3: Saving data...")
    
    if all_shots:
        shots_df = pd.DataFrame(all_shots)
        shots_filename = f'shots_{season.replace("-", "_")}.csv'
        shots_df.to_csv(shots_filename, index=False)
        print(f"Saved {len(all_shots)} shots to {shots_filename}")
        
        # Show shots summary
        if 'event_type' in shots_df.columns:
            shots_by_type = shots_df['event_type'].value_counts()
            print(f"Shots breakdown: {dict(shots_by_type)}")
    
    if all_cards:
        cards_df = pd.DataFrame(all_cards)
        cards_filename = f'cards_{season.replace("-", "_")}.csv'
        cards_df.to_csv(cards_filename, index=False)
        print(f"Saved {len(all_cards)} cards to {cards_filename}")
        
        # Show cards summary
        if 'card_type' in cards_df.columns:
            cards_by_type = cards_df['card_type'].value_counts()
            print(f"Cards breakdown: {dict(cards_by_type)}")
    
    print(f"\nScraping completed!")
    print(f"Successfully processed {success_count}/{len(finished_matches)} matches")
    print(f"Total shots: {len(all_shots)}")
    print(f"Total cards: {len(all_cards)}")


# Example usage
if __name__ == "__main__":
    
        # Configuration for full scraper
        LEAGUE_ID = 47  # Premier League
        SEASON = "2024-2025"
        MAX_MATCHES = 1  # Set to None to scrape all matches, or a number for testing
        
        # Run the full scraper
        scrape_season_data(
            league_id=LEAGUE_ID,
            season=SEASON,
            max_matches=MAX_MATCHES
        )