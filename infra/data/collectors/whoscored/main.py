# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:20:02 2020

@author: aliha
@twitter: rockingAli5 
"""

import warnings
import time
import pandas as pd
pd.options.mode.chained_assignment = None
import json
from bs4 import BeautifulSoup as soup
import re 
from collections import OrderedDict
from datetime import datetime as dt
import itertools
import numpy as np
try:
    from tqdm import trange
except ModuleNotFoundError:
    pass


from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException, ElementClickInterceptedException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# options = webdriver.FirefoxOptions()

# options.add_experimental_option('excludeSwitches', ['enable-logging'])


def dismiss_overlays(driver, wait_time=3):
    """
    Dismiss any overlays/popups that might block interactions on WhoScored
    
    Args:
        driver: Selenium WebDriver instance
        wait_time (int): Time to wait for overlays to appear
    
    Returns:
        bool: True if any overlays were dismissed
    """
    dismissed_any = False
    
    try:
        # Wait a bit for overlays to load
        time.sleep(wait_time)
        
        # Strategy 1: Close cookie consent banners - multiple selectors
        cookie_selectors = [
            "//button[contains(translate(., 'ACCEPT', 'accept'), 'accept')]",
            "//button[contains(@class, 'css-gweyaj')]",
            "//button[@class=' css-gweyaj']",
            "//div[contains(@class, 'qc-cmp2')]//button[contains(., 'AGREE')]",
            "//div[contains(@class, 'qc-cmp2')]//button[contains(., 'CONTINUE')]",
        ]
        
        for selector in cookie_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                for element in elements:
                    if element.is_displayed():
                        try:
                            element.click()
                            print(f"✓ Dismissed overlay using selector: {selector[:50]}...")
                            dismissed_any = True
                            time.sleep(1)
                        except ElementClickInterceptedException:
                            # Try JavaScript click if regular click fails
                            driver.execute_script("arguments[0].click();", element)
                            print(f"✓ Dismissed overlay using JS click")
                            dismissed_any = True
                            time.sleep(1)
            except Exception:
                pass
        
        # Strategy 2: Close any modal overlays or dialogs
        modal_selectors = [
            "//button[contains(@aria-label, 'close')]",
            "//button[contains(@aria-label, 'Close')]",
            "//button[contains(@class, 'close')]",
            "//div[contains(@class, 'enlWCx')]//button",
            "//*[@class='enlWCx']//button",
        ]
        
        for selector in modal_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                for element in elements:
                    if element.is_displayed():
                        try:
                            element.click()
                            print(f"✓ Dismissed modal using selector: {selector[:50]}...")
                            dismissed_any = True
                            time.sleep(1)
                        except ElementClickInterceptedException:
                            driver.execute_script("arguments[0].click();", element)
                            print(f"✓ Dismissed modal using JS click")
                            dismissed_any = True
                            time.sleep(1)
            except Exception:
                pass
        
        # Strategy 3: Remove overlay divs that might be blocking clicks
        try:
            overlay_divs = driver.find_elements(By.XPATH, 
                "//div[contains(@class, 'overlay') or contains(@class, 'modal') or contains(@class, 'popup')]")
            for div in overlay_divs:
                if div.is_displayed():
                    driver.execute_script("arguments[0].remove();", div)
                    print("✓ Removed overlay div")
                    dismissed_any = True
        except Exception:
            pass
            
    except Exception as e:
        print(f"Note: Overlay dismissal attempted: {e}")
    
    if dismissed_any:
        print("✓ Successfully dismissed overlays")
    
    return dismissed_any


TRANSLATE_DICT = {'Jan': 'Jan',
                 'Feb': 'Feb',
                 'Mac': 'Mar',
                 'Apr': 'Apr',
                 'Mei': 'May',
                 'Jun': 'Jun',
                 'Jul': 'Jul',
                 'Ago': 'Aug',
                 'Sep': 'Sep',
                 'Okt': 'Oct',
                 'Nov': 'Nov',
                 'Des': 'Dec',
                 'Jan': 'Jan',
                 'Feb': 'Feb',
                 'Mar': 'Mar',
                 'Apr': 'Apr',
                 'May': 'May',
                 'Jun': 'Jun',
                 'Jul': 'Jul',
                 'Aug': 'Aug',
                 'Sep': 'Sep',
                 'Oct': 'Oct',
                 'Nov': 'Nov',
                 'Dec': 'Dec'}

main_url = 'https://1xbet.whoscored.com/'



def getLeagueUrls(minimize_window=True):
    
    driver = webdriver.Firefox()

    if minimize_window:
        driver.minimize_window()

    driver.get(main_url)
    
    # Dismiss overlays right after loading the page
    dismiss_overlays(driver)
    
    league_names = []
    league_urls = []
    try:
        cookie_button = driver.find_element(By.XPATH, '//*[@class=" css-gweyaj"]').click()
    except NoSuchElementException:
        pass
    tournaments_btn = driver.find_element(By.XPATH, '//*[@id="All-Tournaments-btn"]').click()
    n_button = soup(driver.find_element(By.XPATH, '//*[@id="header-wrapper"]/div/div/div/div[4]/div[2]/div/div/div/div[1]/div/div').get_attribute('innerHTML'), features='lxml').find_all('button')
    n_tournaments = []
    for button in n_button:
        id_button = button.get('id')
        driver.find_element(By.ID, id_button).click()
        n_country = soup(driver.find_element(By.XPATH, '//*[@id="header-wrapper"]/div/div/div/div[4]/div[2]/div/div/div/div[2]').get_attribute('innerHTML'), features='lxml').find_all('div', {'class':'TournamentsDropdownMenu-module_countryDropdownContainer__I9P6n'})

        for country in n_country:
            country_id = country.find('div', {'class': 'TournamentsDropdownMenu-module_countryDropdown__8rtD-'}).get('id')

            # Trouver l'élément avec Selenium et cliquer dessus
            country_element = driver.find_element(By.ID, country_id)
            country_element.click()

            html_tournaments_list = driver.find_element(By.XPATH, '//*[@id="header-wrapper"]/div/div/div/div[4]/div[2]/div/div/div/div[2]').get_attribute('innerHTML')

            # Parse le HTML avec BeautifulSoup pour trouver les liens des tournois
            soup_tournaments = soup(html_tournaments_list, 'html.parser')
            tournaments = soup_tournaments.find_all('a')

            # Ajouter les tournois à la liste n_tournaments
            n_tournaments.extend(tournaments)

            driver.execute_script("arguments[0].click();", country_element)


    for tournament in n_tournaments:
        league_name = tournament.get('href').split('/')[-1]
        league_link = main_url[:-1]+tournament.get('href')
        league_names.append(league_name)
        league_urls.append(league_link)

    leagues = {}
    for name,link in zip(league_names,league_urls):
        leagues[name] = link

    driver.close()
    return leagues


def getMatchUrls(comp_urls, competition, season, maximize_window=True):
    from selenium.webdriver.support.ui import Select
    
    # Run in headless mode
    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')
    
    driver = webdriver.Firefox(options=options)
    
    if maximize_window:
        driver.maximize_window()
    
    comp_url = comp_urls[competition]
    driver.get(comp_url)
    time.sleep(5)
    
    # CRITICAL: Dismiss overlays IMMEDIATELY after page load
    print("Attempting to dismiss overlays...")
    dismiss_overlays(driver, wait_time=2)
    
    # Additional wait after overlay dismissal
    time.sleep(2)
    
    # Wait for seasons dropdown
    try:
        select_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "seasons"))
        )
    except TimeoutException:
        print("ERROR: Could not find seasons dropdown")
        driver.close()
        return []
    
    seasons = driver.find_element(By.XPATH, '//*[@id="seasons"]').get_attribute('innerHTML').split(sep='\n')
    seasons = [i for i in seasons if i]
    
    season_found = False
    
    for i in range(1, len(seasons)+1):
        if driver.find_element(By.XPATH, '//*[@id="seasons"]/option['+str(i)+']').text == season:
            season_found = True
            
            # Use Select class for more reliable dropdown handling
            select = Select(select_element)
            select.select_by_visible_text(season)
            
            # Wait longer for page to fully reload after season change
            time.sleep(8)
            
            # Verify the season actually changed by checking page content
            print(f"Selected season: {season}")
            print(f"Current page URL: {driver.current_url}")
            
            try:
                # Wait for stages element to be present (or determine it doesn't exist)
                try:
                    stages_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "stages"))
                    )
                    stages = stages_element.get_attribute('innerHTML').split(sep='\n')
                    stages = [i for i in stages if i]
                    
                    all_urls = []
                
                    for i in range(1, len(stages)+1):
                        stage_option = driver.find_element(By.XPATH, '//*[@id="stages"]/option['+str(i)+']')
                        stage_text = stage_option.text
                        print(stage_text)
                        
                        should_click = False
                        
                        if competition == 'Champions League' or competition == 'Europa League':
                            if 'Grp' in stage_text or 'Final Stage' in stage_text:
                                should_click = True
                        elif competition == 'Major League Soccer':
                            if 'Grp. ' not in stage_text:
                                should_click = True
                        else:
                            should_click = True
                        
                        if should_click:
                            # Use Select for stages too
                            stage_select = Select(driver.find_element(By.ID, "stages"))
                            stage_select.select_by_visible_text(stage_text)
                            time.sleep(8)
                        
                            driver.execute_script("window.scrollTo(0, 400)")
                            
                            match_urls = getFixtureData(driver)
                            
                            match_urls = getSortedData(match_urls)
                            
                            match_urls2 = [url for url in match_urls if '?' not in url['date'] and '\n' not in url['date']]
                            
                            all_urls += match_urls2
                
                except (NoSuchElementException, TimeoutException):
                    # No stages element - proceed directly to getting fixture data
                    print("No stages dropdown found, getting fixtures directly...")
                    all_urls = []
                    
                    driver.execute_script("window.scrollTo(0, 400)")
                    
                    match_urls = getFixtureData(driver)
                    
                    match_urls = getSortedData(match_urls)
                    
                    match_urls2 = [url for url in match_urls if '?' not in url['date'] and '\n' not in url['date']]
                    
                    all_urls += match_urls2
            
            except Exception as e:
                print(f"Error processing stages: {e}")
                all_urls = []
            
            remove_dup = [dict(t) for t in {tuple(sorted(d.items())) for d in all_urls}]
            all_urls = getSortedData(remove_dup)
            
            driver.close() 
    
            return all_urls
    
    if not season_found:
        season_names = [re.search(r'\>(.*?)\<',season).group(1) for season in seasons]
        driver.close() 
        print('Seasons available: {}'.format(season_names))
        raise ValueError('Season Not Found.')



def getTeamUrls(team, match_urls):
    
    team_data = []
    for fixture in match_urls:
        if fixture['home'] == team or fixture['away'] == team:
            team_data.append(fixture)
    team_data = [a[0] for a in itertools.groupby(team_data)]
                
    return team_data


def getMatchesData(match_urls, minimize_window=True):
    
    matches = []
    
    driver = webdriver.Firefox()
    if minimize_window:
        driver.minimize_window()
    
    try:
        for i in trange(len(match_urls), desc='Getting Match Data'):
            # recommended to avoid getting blocked by incapsula/imperva bots
            time.sleep(7)
            match_data = getMatchData(driver, main_url+match_urls[i]['url'], display=False, close_window=False)
            match_data['home_team'] = match_urls[i]['home']
            match_data['away_team'] = match_urls[i]['away']
            matches.append(match_data)
    except NameError:
        print('Recommended: \'pip install tqdm\' for a progress bar while the data gets scraped....')
        time.sleep(7)
        for i in range(len(match_urls)):
            match_data = getMatchData(driver, main_url+match_urls[i]['url'], display=False, close_window=False)
            matches.append(match_data)
    
    driver.close()
    
    return matches




def getFixtureData(driver):
    """Get fixture data with robust overlay handling"""
    matches_ls = []
    iteration_count = 0
    max_iterations = 100  # Safety limit
    
    while iteration_count < max_iterations:
        iteration_count += 1
        initial = driver.page_source
        
        all_fixtures = driver.find_elements(By.CLASS_NAME, 'Accordion-module_accordion__UuHD0')
        for dates in all_fixtures:
            fixtures = dates.find_elements(By.CLASS_NAME, 'Match-module_row__zwBOn')
            date_row = dates.find_element(By.CLASS_NAME, 'Accordion-module_header__HqzWD')
            for row in fixtures:
                url = row.find_element(By.TAG_NAME, 'a')
                if 'live' in url.get_attribute('href'):
                    match_dict = {}
                    element = soup(row.get_attribute('innerHTML'), features='lxml')
                    teams_tag = element.find("div", {"class":"Match-module_teams__sGVeq"})
                    link_tag = element.find("a")
                    match_dict['date'] = date_row.text
                    match_dict['home'] = teams_tag.find_all('a')[0].text
                    match_dict['away'] = teams_tag.find_all('a')[1].text
                    match_dict['score'] = ':'.join([t.text for t in link_tag.find_all('span')])
                    match_dict['url'] = link_tag['href']
                    matches_ls.append(match_dict)
        
        # Try to click the previous button with multiple strategies
        try:
            prev_btn = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, 'dayChangeBtn-prev'))
            )
            
            # Strategy 1: Regular click
            try:
                prev_btn.click()
                time.sleep(2)
            except ElementClickInterceptedException:
                # Strategy 2: Scroll into view and click
                try:
                    driver.execute_script("arguments[0].scrollIntoView(true);", prev_btn)
                    time.sleep(1)
                    prev_btn.click()
                    time.sleep(2)
                except ElementClickInterceptedException:
                    # Strategy 3: Dismiss overlays and try again
                    print("Button blocked, dismissing overlays...")
                    dismiss_overlays(driver, wait_time=1)
                    time.sleep(1)
                    try:
                        prev_btn.click()
                        time.sleep(2)
                    except ElementClickInterceptedException:
                        # Strategy 4: JavaScript click
                        print("Using JavaScript click...")
                        driver.execute_script("arguments[0].click();", prev_btn)
                        time.sleep(2)
            
            final = driver.page_source
            if initial == final:
                print(f"Reached end of fixtures after {iteration_count} iterations")
                break
                
        except TimeoutException:
            print("Previous button not found, ending iteration")
            break
        except Exception as e:
            print(f"Error during pagination: {e}")
            break

    print(f"Collected {len(matches_ls)} total fixtures")
    return matches_ls






def translateDate(data):
    
    unwanted = []
    for match in data:
        date = match['date'].split()
        if '?' not in date[0]:
            try:
                match['date'] = ' '.join([TRANSLATE_DICT[date[0]], date[1], date[2]])
            except KeyError:
                print(date)
        else:
            unwanted.append(data.index(match))
    
    # remove matches that got suspended/postponed
    for i in sorted(unwanted, reverse = True):
        del data[i]
    
    return data


def getSortedData(data):
    data = sorted(data, key = lambda i: dt.strptime(i['date'], '%A, %b %d %Y'))
    return data
    



def getMatchData(driver, url, display=True, close_window=True):
    try:
        driver.get(url)
    except WebDriverException:
        driver.get(url)

    time.sleep(5)
    # get script data from page source
    script_content = driver.find_element(By.XPATH, '//*[@id="layout-wrapper"]/script[1]').get_attribute('innerHTML')


    # clean script content
    script_content = re.sub(r"[\n\t]*", "", script_content)
    script_content = script_content[script_content.index("matchId"):script_content.rindex("}")]


    # this will give script content in list form 
    script_content_list = list(filter(None, script_content.strip().split(',            ')))
    metadata = script_content_list.pop(1) 


    # string format to json format
    match_data = json.loads(metadata[metadata.index('{'):])
    keys = [item[:item.index(':')].strip() for item in script_content_list]
    values = [item[item.index(':')+1:].strip() for item in script_content_list]
    for key,val in zip(keys, values):
        match_data[key] = json.loads(val)


    # get other details about the match
    region = driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/span[1]').text
    league = driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - ')[0]
    season = driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - ')[1]
    if len(driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - ')) == 2:
        competition_type = 'League'
        competition_stage = ''
    elif len(driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - '))== 3:
        competition_type = 'Knock Out'
        competition_stage = driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - ')[-1]
    else:
        print('Getting more than 3 types of information about the competition.')

    match_data['region'] = region
    match_data['league'] = league
    match_data['season'] = season
    match_data['competitionType'] = competition_type
    match_data['competitionStage'] = competition_stage


    # sort match_data dictionary alphabetically
    match_data = OrderedDict(sorted(match_data.items()))
    match_data = dict(match_data)
    if display:
        print('Region: {}, League: {}, Season: {}, Match Id: {}'.format(region, league, season, match_data['matchId']))
    
    
    if close_window:
        driver.close()
        
    return match_data





def createEventsDF(data):
    events = data['events']
    for event in events:
        event.update({'matchId' : data['matchId'],
                        'startDate' : data['startDate'],
                        'startTime' : data['startTime'],
                        'score' : data['score'],
                        'ftScore' : data['ftScore'],
                        'htScore' : data['htScore'],
                        'etScore' : data['etScore'],
                        'venueName' : data['venueName'],
                        'maxMinute' : data['maxMinute'],
                        'homeTeam' : data['home_team'],
                        'awayTeam' : data['away_team']})
    events_df = pd.DataFrame(events)

    # clean period column
    events_df['period'] = pd.json_normalize(events_df['period'])['displayName']

    # clean type column
    events_df['type'] = pd.json_normalize(events_df['type'])['displayName']

    # clean outcomeType column
    events_df['outcomeType'] = pd.json_normalize(events_df['outcomeType'])['displayName']

    # clean outcomeType column
    try:
        x = events_df['cardType'].fillna({i: {} for i in events_df.index})
        events_df['cardType'] = pd.json_normalize(x)['displayName'].fillna(False)
    except KeyError:
        events_df['cardType'] = False

    eventTypeDict = data['matchCentreEventTypeJson']  
    events_df['satisfiedEventsTypes'] = events_df['satisfiedEventsTypes'].apply(lambda x: [list(eventTypeDict.keys())[list(eventTypeDict.values()).index(event)] for event in x])

    # clean qualifiers column
    try:
        for i in events_df.index:
            row = events_df.loc[i, 'qualifiers'].copy()
            if len(row) != 0:
                for irow in range(len(row)):
                    row[irow]['type'] = row[irow]['type']['displayName']
    except TypeError:
        pass


    # clean isShot column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if 'isShot' in events_df.columns:
            events_df['isShot'] = events_df['isShot'].replace(np.nan, False).infer_objects(copy=False)
        else:
            events_df['isShot'] = False

        # clean isGoal column
        if 'isGoal' in events_df.columns:
            events_df['isGoal'] = events_df['isGoal'].replace(np.nan, False).infer_objects(copy=False)
        else:
            events_df['isGoal'] = False

    # add player name column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        events_df.loc[events_df.playerId.notna(), 'playerId'] = events_df.loc[events_df.playerId.notna(), 'playerId'].astype(int).astype(str)    
    player_name_col = events_df.loc[:, 'playerId'].map(data['playerIdNameDictionary']) 
    events_df.insert(loc=events_df.columns.get_loc("playerId")+1, column='playerName', value=player_name_col)

    # add home/away column
    h_a_col = events_df['teamId'].map({data['home']['teamId']:'h', data['away']['teamId']:'a'})
    events_df.insert(loc=events_df.columns.get_loc("teamId")+1, column='h_a', value=h_a_col)


    # adding shot body part column
    events_df['shotBodyType'] =  np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for i in events_df.loc[events_df.isShot==True].index:
            for j in events_df.loc[events_df.isShot==True].qualifiers.loc[i]:
                if j['type'] == 'RightFoot' or j['type'] == 'LeftFoot' or j['type'] == 'Head' or j['type'] == 'OtherBodyPart':
                    events_df.loc[i, 'shotBodyType'] = j['type']


    # adding shot situation column
    events_df['situation'] =  np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for i in events_df.loc[events_df.isShot==True].index:
            for j in events_df.loc[events_df.isShot==True].qualifiers.loc[i]:
                if j['type'] == 'FromCorner' or j['type'] == 'SetPiece' or j['type'] == 'DirectFreekick':
                    events_df.loc[i, 'situation'] = j['type']
                if j['type'] == 'RegularPlay':
                    events_df.loc[i, 'situation'] = 'OpenPlay' 

    event_types = list(data['matchCentreEventTypeJson'].keys())
    event_type_cols = pd.DataFrame({event_type: pd.Series([event_type in row for row in events_df['satisfiedEventsTypes']]) for event_type in event_types})
    events_df = pd.concat([events_df, event_type_cols], axis=1)


    return events_df
    



def createMatchesDF(data):
    columns_req_ls = ['matchId', 'attendance', 'venueName', 'startTime', 'startDate',
                      'score', 'home', 'away', 'referee']
    matches_df = pd.DataFrame(columns=columns_req_ls)
    if type(data) == dict:
        matches_dict = dict([(key,val) for key,val in data.items() if key in columns_req_ls])
        matches_df = pd.DataFrame(matches_dict, columns=columns_req_ls).reset_index(drop=True)
        matches_df[['home', 'away']] = np.nan  
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            matches_df['home'].iloc[0] = [data['home']]
            matches_df['away'].iloc[0] = [data['away']]
    else:
        for match in data:
            matches_dict = dict([(key,val) for key,val in match.items() if key in columns_req_ls])
            matches_df = pd.DataFrame(matches_dict, columns=columns_req_ls).reset_index(drop=True)
    
    matches_df = matches_df.set_index('matchId')        
    return matches_df




def load_EPV_grid(fname=r'/Users/admin/Documents/dev/algobetting/infra/data/collectors/whoscored/EPV_grid.csv'):
    """ load_EPV_grid(fname='EPV_grid.csv')
    
    # load pregenerated EPV surface from file. 
    
    Parameters
    -----------
        fname: filename & path of EPV grid (default is 'EPV_grid.csv' in the curernt directory)
        
    Returns
    -----------
        EPV: The EPV surface (default is a (32,50) grid)
    
    """
    epv = np.loadtxt(fname, delimiter=',')
    return epv






def get_EPV_at_location(position,EPV,attack_direction,field_dimen=(106.,68.)):
    """ get_EPV_at_location
    
    Returns the EPV value at a given (x,y) location
    
    Parameters
    -----------
        position: Tuple containing the (x,y) pitch position
        EPV: tuple Expected Possession value grid (loaded using load_EPV_grid() )
        attack_direction: Sets the attack direction (1: left->right, -1: right->left)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
            
    Returrns
    -----------
        EPV value at input position
        
    """
    
    x,y = position
    if abs(x)>field_dimen[0]/2. or abs(y)>field_dimen[1]/2.:
        return 0.0 # Position is off the field, EPV is zero
    else:
        if attack_direction==-1:
            EPV = np.fliplr(EPV)
        ny,nx = EPV.shape
        dx = field_dimen[0]/float(nx)
        dy = field_dimen[1]/float(ny)
        ix = (x+field_dimen[0]/2.-0.0001)/dx
        iy = (y+field_dimen[1]/2.-0.0001)/dy
        return EPV[int(iy),int(ix)]



                

def to_metric_coordinates_from_whoscored(data,field_dimen=(106.,68.) ):
    '''
    Convert positions from Whoscored units to meters (with origin at centre circle)
    '''
    x_columns = [c for c in data.columns if c[-1].lower()=='x'][:2]
    y_columns = [c for c in data.columns if c[-1].lower()=='y'][:2]
    x_columns_mod = [c+'_metrica' for c in x_columns]
    y_columns_mod = [c+'_metrica' for c in y_columns]
    data[x_columns_mod] = (data[x_columns]/100*106)-53
    data[y_columns_mod] = (data[y_columns]/100*68)-34
    return data




def addEpvToDataFrame(data):

    # loading EPV data
    EPV = load_EPV_grid(r'/Users/admin/Documents/dev/algobetting/infra/data/collectors/whoscored/EPV_grid.csv')

    # converting opta coordinates to metric coordinates
    data = to_metric_coordinates_from_whoscored(data)

    # calculating EPV for events
    EPV_difference = []
    for i in data.index:
        if data.loc[i, 'type'] == 'Pass' and data.loc[i, 'outcomeType'] == 'Successful':
            start_pos = (data.loc[i, 'x_metrica'], data.loc[i, 'y_metrica'])
            start_epv = get_EPV_at_location(start_pos, EPV, attack_direction=1)
            
            end_pos = (data.loc[i, 'endX_metrica'], data.loc[i, 'endY_metrica'])
            end_epv = get_EPV_at_location(end_pos, EPV, attack_direction=1)
            
            diff = end_epv - start_epv
            EPV_difference.append(diff)
            
        else:
            EPV_difference.append(np.nan)
    
    data = data.assign(EPV_difference = EPV_difference)
    
    
    # dump useless columns
    drop_cols = ['x_metrica', 'endX_metrica', 'y_metrica',
                 'endY_metrica']
    data.drop(drop_cols, axis=1, inplace=True)
    data.rename(columns={'EPV_difference': 'EPV'}, inplace=True)
    
    return data