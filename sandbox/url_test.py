import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime

def extract_match_info_from_url(url):
    """Extract detailed match information from a FotMob URL"""
    print(f"\n=== ANALYZING URL: {url} ===")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        script = soup.find('script', {'id': '__NEXT_DATA__'})
        
        if not script or not script.string:
            print("❌ No __NEXT_DATA__ script found")
            return None
        
        # Parse JSON
        json_string = script.string
        before, sep, after = json_string.partition(',"seo"')
        before = before.replace('{"props":{"pageProps":', '')
        before = before + '}'
        
        try:
            data = json.loads(before)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            return None
        
        # Extract match details
        match_info = {}
        
        # Try to find general match info
        general = data.get('general', {})
        if general:
            match_info['home_team'] = general.get('homeTeam', {}).get('name', 'Unknown')
            match_info['away_team'] = general.get('awayTeam', {}).get('name', 'Unknown')
            match_info['home_score'] = general.get('homeTeam', {}).get('score')
            match_info['away_score'] = general.get('awayTeam', {}).get('score')
            match_info['match_id'] = general.get('matchId')
            match_info['parent_league_id'] = general.get('parentLeagueId')
            match_info['league_id'] = general.get('leagueId')
            match_info['status'] = general.get('status', {}).get('reason')
            match_info['kickoff_time'] = general.get('status', {}).get('utcTime')
        
        # Extract additional identifiers that might be relevant
        if 'content' in data:
            content = data['content']
            # Look for any other match identifiers
            if 'matchFacts' in content:
                match_facts = content['matchFacts']
                if 'matchId' in match_facts:
                    match_info['match_facts_id'] = match_facts['matchId']
        
        # Extract shots info for verification
        shots_count = 0
        if 'content' in data and 'shotmap' in data['content'] and 'shots' in data['content']['shotmap']:
            shots_count = len(data['content']['shotmap']['shots'])
        match_info['shots_count'] = shots_count
        
        # Extract events info
        events_count = 0
        red_cards_count = 0
        if 'content' in data and 'matchFacts' in data['content'] and 'events' in data['content']['matchFacts']:
            events = data['content']['matchFacts']['events'].get('events', [])
            events_count = len(events)
            red_cards_count = sum(1 for e in events if e.get('card') == 'Red')
        match_info['events_count'] = events_count
        match_info['red_cards_count'] = red_cards_count
        
        return match_info
        
    except Exception as e:
        print(f"❌ Error analyzing URL: {e}")
        return None

def compare_match_data(url1, url2):
    """Compare match data from two URLs to see if they're actually the same match"""
    print(f"=== COMPARING MATCH DATA ===")
    
    info1 = extract_match_info_from_url(url1)
    info2 = extract_match_info_from_url(url2)
    
    if not info1 or not info2:
        print("❌ Could not extract data from one or both URLs")
        return
    
    print(f"\n--- URL 1 Match Info ---")
    for key, value in info1.items():
        print(f"  {key}: {value}")
    
    print(f"\n--- URL 2 Match Info ---")
    for key, value in info2.items():
        print(f"  {key}: {value}")
    
    print(f"\n--- COMPARISON RESULTS ---")
    same_match = True
    
    # Compare key fields
    comparison_fields = [
        'home_team', 'away_team', 'home_score', 'away_score', 
        'match_id', 'parent_league_id', 'league_id', 'kickoff_time',
        'shots_count', 'events_count', 'red_cards_count'
    ]
    
    for field in comparison_fields:
        val1 = info1.get(field)
        val2 = info2.get(field)
        
        if val1 == val2:
            print(f"  ✅ {field}: SAME ({val1})")
        else:
            print(f"  ❌ {field}: DIFFERENT (URL1: {val1}, URL2: {val2})")
            same_match = False
    
    print(f"\n🎯 CONCLUSION: {'Same match' if same_match else 'DIFFERENT MATCHES!'}")
    return same_match, info1, info2

def analyze_fragment_importance(base_url, fragment1, fragment2):
    """Test if the fragment part of the URL matters"""
    print(f"\n=== TESTING FRAGMENT IMPORTANCE ===")
    
    test_urls = [
        f"{base_url}#{fragment1}",
        f"{base_url}#{fragment2}",
        base_url,  # No fragment
        f"{base_url}#12345",  # Random fragment
    ]
    
    results = {}
    
    for url in test_urls:
        print(f"\nTesting: {url}")
        info = extract_match_info_from_url(url)
        if info:
            # Create a simple signature for the match
            signature = f"{info.get('home_team', 'Unknown')} vs {info.get('away_team', 'Unknown')} - Score: {info.get('home_score', '?')}-{info.get('away_score', '?')}"
            results[url] = signature
            print(f"  Result: {signature}")
        else:
            results[url] = "Failed to load"
            print(f"  Result: Failed to load")
    
    # Check if fragment matters
    unique_results = set(results.values())
    if len(unique_results) == 1:
        print(f"\n🎯 FRAGMENT ANALYSIS: Fragment ID does NOT matter - all URLs show same match")
    else:
        print(f"\n🎯 FRAGMENT ANALYSIS: Fragment ID MATTERS - different fragments show different matches!")
        print("Unique match results:")
        for i, result in enumerate(unique_results, 1):
            print(f"  {i}. {result}")

def investigate_url_generation_pattern():
    """Try to understand the URL pattern"""
    print(f"\n=== URL PATTERN INVESTIGATION ===")
    print("Based on the analysis above, here are possible explanations:")
    print()
    print("1. FRAGMENT AS MATCH INSTANCE ID:")
    print("   - The fragment (#4506263, #4813391) might represent different 'instances' of the same match")
    print("   - Could be different rounds, seasons, or match states")
    print()
    print("2. TIME-BASED ROUTING:")
    print("   - FotMob might generate different URLs based on when/how you access the match")
    print("   - Fragment could be timestamp-based or session-based")
    print()
    print("3. LEAGUE/COMPETITION CONTEXT:")
    print("   - Same teams might play multiple times (league, cup, etc.)")
    print("   - Fragment could distinguish between different competitions")
    print()
    print("4. URL REDIRECT/ALIAS SYSTEM:")
    print("   - FotMob might use multiple URLs pointing to the same match")
    print("   - Server-side routing based on fragment")

def generate_robust_url_strategy():
    """Generate a strategy for handling these URL variations"""
    return """
=== ROBUST URL HANDLING STRATEGY ===

Based on the analysis, here's a recommended approach:

1. PRESERVE ORIGINAL URLS:
   - Don't modify the URLs extracted from the match listing
   - The fragment ID seems to be important for routing

2. VALIDATION APPROACH:
   def validate_and_extract_data(match_url, expected_home_team, expected_away_team):
       # Get data from URL
       data = scrape_match_details(match_url, expected_home_team, expected_away_team)
       
       # Validate that we got the right match
       if data and data.get('match_info'):
           actual_home = data['match_info'].get('home_team', '').lower()
           actual_away = data['match_info'].get('away_team', '').lower()
           expected_home = expected_home_team.lower()
           expected_away = expected_away_team.lower()
           
           # Check if teams match (allowing for name variations)
           if (expected_home in actual_home or actual_home in expected_home) and \
              (expected_away in actual_away or actual_away in expected_away):
               return data  # Correct match
           else:
               print(f"WARNING: Team mismatch! Expected: {expected_home_team} vs {expected_away_team}")
               print(f"         Got: {data['match_info'].get('home_team')} vs {data['match_info'].get('away_team')}")
               return None
       
       return data

3. FALLBACK STRATEGY:
   - If original URL fails, try variations systematically
   - Log all attempts for debugging
   - Always validate team names match expectations

4. DEBUG LOGGING:
   - Log the exact URL being processed
   - Log the team names extracted from the page
   - Log any mismatches for manual review
"""

# Main diagnostic execution
if __name__ == "__main__":
    # Test the two problematic URLs
    url1 = "https://www.fotmob.com/matches/fulham-vs-manchester-united/3cqww9#4506263"
    url2 = "https://www.fotmob.com/en-GB/matches/fulham-vs-manchester-united/3cqww9#4813391"
    
    # Compare the match data
    same_match, info1, info2 = compare_match_data(url1, url2)
    
    # Test fragment importance
    base_url = "https://www.fotmob.com/matches/fulham-vs-manchester-united/3cqww9"
    analyze_fragment_importance(base_url, "4506263", "4813391")
    
    # Investigate patterns
    investigate_url_generation_pattern()
    
    # Generate strategy
    print(generate_robust_url_strategy())