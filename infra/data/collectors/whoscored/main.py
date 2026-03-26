# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:20:02 2020

@author: aliha
@twitter: rockingAli5 

UPDATED: Fixed overlay/popup blocking issues with safe_click and enhanced dismiss_overlays
UPDATED: Fixed pandas 2.x compatibility issues
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
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import NoSuchElementException, WebDriverException, ElementClickInterceptedException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def create_driver_with_options(headless=False, minimize=False):
    options = Options()
    options.set_preference("dom.webnotifications.enabled", False)
    options.set_preference("dom.push.enabled", False)
    options.set_preference("dom.webnotifications.serviceworker.enabled", False)
    options.set_preference("dom.serviceWorkers.enabled", False)
    
    if headless:
        options.add_argument('--headless')
    
    driver = webdriver.Firefox(options=options)
    
    if minimize and not headless:
        driver.minimize_window()
    
    return driver


def safe_click(driver, element, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            element.click()
            return True
        except ElementClickInterceptedException:
            print(f"⚠️  Click blocked by overlay (attempt {attempt + 1}/{max_attempts})")
            dismiss_overlays(driver, wait_time=1)
            time.sleep(1)
            if attempt == max_attempts - 1:
                print("💡 Trying JavaScript click as fallback...")
                try:
                    driver.execute_script("arguments[0].click();", element)
                    print("✅ JavaScript click succeeded")
                    return True
                except Exception as e:
                    print(f"❌ JavaScript click failed: {e}")
                    return False
        except Exception as e:
            print(f"❌ Unexpected error during click: {e}")
            if attempt == max_attempts - 1:
                return False
            time.sleep(1)
    
    return False


def dismiss_overlays(driver, wait_time=3):
    dismissed_any = False
    
    try:
        time.sleep(wait_time)
        
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
                            driver.execute_script("arguments[0].click();", element)
                            print(f"✓ Dismissed overlay using JS click")
                            dismissed_any = True
                            time.sleep(1)
            except Exception:
                pass
        
        modal_selectors = [
            "//button[contains(@aria-label, 'close')]",
            "//button[contains(@aria-label, 'Close')]",
            "//button[contains(@class, 'close')]",
            "//div[contains(@class, 'enlWCx')]//button",
            "//*[@class='enlWCx']//button",
            "//button[contains(@class, 'webpush-swal2-close')]",
            "//button[contains(@class, 'swal2-close')]",
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
        
        try:
            overlay_divs = driver.find_elements(By.XPATH, 
                "//div[contains(@class, 'overlay') or contains(@class, 'modal') or contains(@class, 'popup') or contains(@class, 'webpush')]")
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
                 'Mar': 'Mar',
                 'May': 'May',
                 'Aug': 'Aug',
                 'Oct': 'Oct',
                 'Dec': 'Dec'}

main_url = 'https://1xbet.whoscored.com/'


def getLeagueUrls(minimize_window=True):
    driver = create_driver_with_options(minimize=minimize_window)
    driver.get(main_url)
    time.sleep(3)
    dismiss_overlays(driver, wait_time=2)
    
    league_names = []
    league_urls = []
    
    try:
        cookie_button = driver.find_element(By.XPATH, '//*[@class=" css-gweyaj"]')
        safe_click(driver, cookie_button)
    except NoSuchElementException:
        pass
    
    try:
        tournaments_btn = driver.find_element(By.XPATH, '//*[@id="All-Tournaments-btn"]')
        if not safe_click(driver, tournaments_btn):
            print("❌ Failed to click tournaments button")
            driver.close()
            return {}
    except NoSuchElementException:
        print("❌ Tournaments button not found")
        driver.close()
        return {}
    
    time.sleep(1)
    
    n_button = soup(driver.find_element(By.XPATH, '//*[@id="header-wrapper"]/div/div/div/div[4]/div[2]/div/div/div/div[1]/div/div').get_attribute('innerHTML'), features='lxml').find_all('button')
    n_tournaments = []
    
    for button in n_button:
        id_button = button.get('id')
        try:
            button_element = driver.find_element(By.ID, id_button)
            
            if not safe_click(driver, button_element):
                print(f"⚠️  Failed to click button: {id_button}")
                continue
            
            time.sleep(0.5)
            
            n_country = soup(driver.find_element(By.XPATH, '//*[@id="header-wrapper"]/div/div/div/div[4]/div[2]/div/div/div/div[2]').get_attribute('innerHTML'), features='lxml').find_all('div', {'class':'TournamentsDropdownMenu-module_countryDropdownContainer__I9P6n'})

            for country in n_country:
                country_id = country.find('div', {'class': 'TournamentsDropdownMenu-module_countryDropdown__8rtD-'}).get('id')
                country_element = driver.find_element(By.ID, country_id)
                driver.execute_script("arguments[0].click();", country_element)
                time.sleep(0.5)

                html_tournaments_list = driver.find_element(By.XPATH, '//*[@id="header-wrapper"]/div/div/div/div[4]/div[2]/div/div/div/div[2]').get_attribute('innerHTML')
                soup_tournaments = soup(html_tournaments_list, 'html.parser')
                tournaments = soup_tournaments.find_all('a')
                n_tournaments.extend(tournaments)
                driver.execute_script("arguments[0].click();", country_element)
                time.sleep(0.3)
                
        except Exception as e:
            print(f"⚠️  Error processing button {id_button}: {e}")
            continue

    for tournament in n_tournaments:
        league_name = tournament.get('href').split('/')[-1]
        league_link = main_url[:-1]+tournament.get('href')
        league_names.append(league_name)
        league_urls.append(league_link)

    leagues = {}
    for name, link in zip(league_names, league_urls):
        leagues[name] = link

    driver.close()
    return leagues


PL_URL = 'https://1xbet.whoscored.com/regions/252/tournaments/2/england-premier-league'


def getMatchUrls(season, start_date=None, end_date=None):
    """
    Fetch Premier League fixture URLs for a given season.
    Pass start_date to stop paginating backwards once we've gone past it —
    avoids scraping the entire season when you only need a small window.

    Args:
        season (str)          : e.g. '2025/2026'
        start_date (datetime) : stop paginating backwards before this date
        end_date (datetime)   : unused for pagination, kept for API consistency
    """
    from selenium.webdriver.support.ui import Select

    driver = create_driver_with_options(headless=True)
    driver.get(PL_URL)
    time.sleep(3)

    print("Attempting to dismiss overlays...")
    dismiss_overlays(driver, wait_time=1)

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

    for i in range(1, len(seasons) + 1):
        if driver.find_element(By.XPATH, f'//*[@id="seasons"]/option[{i}]').text == season:
            select = Select(select_element)
            select.select_by_visible_text(season)
            time.sleep(5)

            print(f"Selected season: {season}")

            driver.execute_script("window.scrollTo(0, 400)")
            match_urls = getFixtureData(driver, stop_before=start_date)
            match_urls = getSortedData(match_urls)
            all_urls = [url for url in match_urls if '?' not in url['date'] and '\n' not in url['date']]

            remove_dup = [dict(t) for t in {tuple(sorted(d.items())) for d in all_urls}]
            all_urls = getSortedData(remove_dup)

            driver.close()
            return all_urls

    season_names = [re.search(r'\>(.*?)\<', s).group(1) for s in seasons]
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
    driver = create_driver_with_options(minimize=minimize_window)
    try:
        for i in trange(len(match_urls), desc='Getting Match Data'):
            time.sleep(7)
            try:
                match_data = getMatchData(driver, main_url+match_urls[i]['url'], display=False, close_window=False)
                match_data['home_team'] = match_urls[i]['home']
                match_data['away_team'] = match_urls[i]['away']
                matches.append(match_data)
            except ValueError as e:
                print(f"\nSkipping match {match_urls[i].get('url', i)}: {e}")
                continue
    except NameError:
        print('Recommended: \'pip install tqdm\' for a progress bar while the data gets scraped....')
        for i in range(len(match_urls)):
            time.sleep(7)
            try:
                match_data = getMatchData(driver, main_url+match_urls[i]['url'], display=False, close_window=False)
                match_data['home_team'] = match_urls[i]['home']
                match_data['away_team'] = match_urls[i]['away']
                matches.append(match_data)
            except ValueError as e:
                print(f"\nSkipping match {match_urls[i].get('url', i)}: {e}")
                continue
    driver.close()
    return matches


def getFixtureData(driver, stop_before=None):
    """
    Paginate backwards through fixtures, collecting match data.

    Args:
        driver      : Selenium WebDriver
        stop_before : datetime — stop as soon as every date on the current page
                      is earlier than this date. Pass start_date here to avoid
                      scraping the whole season when you only need a small window.
    """
    matches_ls = []
    iteration_count = 0
    max_iterations = 100

    while iteration_count < max_iterations:
        iteration_count += 1
        initial = driver.page_source

        all_fixtures = driver.find_elements(By.CLASS_NAME, 'Accordion-module_accordion__UuHD0')

        page_dates = []
        for dates in all_fixtures:
            fixtures = dates.find_elements(By.CLASS_NAME, 'Match-module_row__zwBOn')
            date_row = dates.find_element(By.CLASS_NAME, 'Accordion-module_header__HqzWD')
            date_text = date_row.text

            # Try to parse the date for early-exit logic
            if stop_before and date_text and '?' not in date_text and '\n' not in date_text:
                try:
                    page_dates.append(dt.strptime(date_text, '%A, %b %d %Y'))
                except ValueError:
                    pass

            for row in fixtures:
                url = row.find_element(By.TAG_NAME, 'a')
                if 'live' in url.get_attribute('href'):
                    match_dict = {}
                    element = soup(row.get_attribute('innerHTML'), features='lxml')
                    teams_tag = element.find("div", {"class": "Match-module_teams__sGVeq"})
                    link_tag = element.find("a")
                    match_dict['date'] = date_text
                    match_dict['home'] = teams_tag.find_all('a')[0].text
                    match_dict['away'] = teams_tag.find_all('a')[1].text
                    match_dict['score'] = ':'.join([t.text for t in link_tag.find_all('span')])
                    match_dict['url'] = link_tag['href']
                    matches_ls.append(match_dict)

        # Early exit: all dates on this page are before our window
        if stop_before and page_dates and all(d < stop_before for d in page_dates):
            print(f"✓ Reached dates before {stop_before.date()} — stopping pagination early")
            break

        try:
            prev_btn = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, 'dayChangeBtn-prev'))
            )

            if not safe_click(driver, prev_btn):
                print("Could not click previous button, ending pagination")
                break

            time.sleep(1)

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

    print(f"Collected {len(matches_ls)} fixtures in {iteration_count} iterations")
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
    
    for i in sorted(unwanted, reverse=True):
        del data[i]
    
    return data


def getSortedData(data):
    data = sorted(data, key=lambda i: dt.strptime(i['date'], '%A, %b %d %Y'))
    return data


def getMatchData(driver, url, display=True, close_window=True, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            driver.get(url)
        except WebDriverException:
            driver.get(url)

        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="layout-wrapper"]/script[1]'))
            )
            break  # page loaded successfully
        except TimeoutException:
            print(f"\u26a0\ufe0f  Page did not load properly (attempt {attempt}/{max_retries}): {url}")
            if attempt == max_retries:
                raise RuntimeError(f"Failed to load match page after {max_retries} attempts: {url}")
            time.sleep(5 * attempt)

    script_content = driver.find_element(By.XPATH, '//*[@id="layout-wrapper"]/script[1]').get_attribute('innerHTML')
    script_content = re.sub(r"[\n\t]*", "", script_content)
    script_content = script_content[script_content.index("matchId"):script_content.rindex("}")]
    script_content_list = list(filter(None, script_content.strip().split(',            ')))
    metadata = script_content_list.pop(1) 
    match_data = json.loads(metadata[metadata.index('{'):])
    keys = [item[:item.index(':')].strip() for item in script_content_list]
    values = [item[item.index(':')+1:].strip() for item in script_content_list]
    for key, val in zip(keys, values):
        match_data[key] = json.loads(val)

    region = driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/span[1]').text
    league = driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - ')[0]
    season = driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - ')[1]
    if len(driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - ')) == 2:
        competition_type = 'League'
        competition_stage = ''
    elif len(driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - ')) == 3:
        competition_type = 'Knock Out'
        competition_stage = driver.find_element(By.XPATH, '//*[@id="breadcrumb-nav"]/a').text.split(' - ')[-1]
    else:
        print('Getting more than 3 types of information about the competition.')

    match_data['region'] = region
    match_data['league'] = league
    match_data['season'] = season
    match_data['competitionType'] = competition_type
    match_data['competitionStage'] = competition_stage

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

    # clean cardType column
    try:
        x = events_df['cardType'].fillna({i: {} for i in events_df.index})
        cardType_values = pd.json_normalize(x)['displayName'].fillna(False)
        events_df['cardType'] = cardType_values.values
    except KeyError:
        events_df['cardType'] = False

    eventTypeDict = data['matchCentreEventTypeJson']  
    events_df['satisfiedEventsTypes'] = events_df['satisfiedEventsTypes'].apply(
        lambda x: [list(eventTypeDict.keys())[list(eventTypeDict.values()).index(event)] for event in x]
    )

    # clean qualifiers column
    try:
        for i in events_df.index:
            row = events_df.loc[i, 'qualifiers'].copy()
            if len(row) != 0:
                for irow in range(len(row)):
                    row[irow]['type'] = row[irow]['type']['displayName']
    except TypeError:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if 'isShot' in events_df.columns:
            events_df['isShot'] = events_df['isShot'].fillna(False).astype(bool)
        else:
            events_df['isShot'] = False

        if 'isGoal' in events_df.columns:
            events_df['isGoal'] = events_df['isGoal'].fillna(False).astype(bool)
        else:
            events_df['isGoal'] = False

    events_df['playerId'] = events_df['playerId'].apply(
        lambda x: str(int(x)) if pd.notna(x) else x
    )

    player_name_col = events_df['playerId'].map(data['playerIdNameDictionary']) 
    events_df.insert(loc=events_df.columns.get_loc("playerId")+1, column='playerName', value=player_name_col)

    h_a_col = events_df['teamId'].map({data['home']['teamId']: 'h', data['away']['teamId']: 'a'})
    events_df.insert(loc=events_df.columns.get_loc("teamId")+1, column='h_a', value=h_a_col)

    events_df['shotBodyType'] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        shot_index = events_df.loc[events_df['isShot'] == True].index
        for i in shot_index:
            for j in events_df.loc[i, 'qualifiers']:
                if j['type'] in ('RightFoot', 'LeftFoot', 'Head', 'OtherBodyPart'):
                    events_df.loc[i, 'shotBodyType'] = j['type']

    events_df['situation'] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for i in shot_index:
            for j in events_df.loc[i, 'qualifiers']:
                if j['type'] in ('FromCorner', 'SetPiece', 'DirectFreekick'):
                    events_df.loc[i, 'situation'] = j['type']
                if j['type'] == 'RegularPlay':
                    events_df.loc[i, 'situation'] = 'OpenPlay'

    event_types = list(data['matchCentreEventTypeJson'].keys())
    event_type_cols = pd.DataFrame(
        {event_type: pd.array(
            [event_type in row for row in events_df['satisfiedEventsTypes']], dtype='boolean'
        ) for event_type in event_types}
    )
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

        matches_df.at[0, 'home'] = [data['home']]
        matches_df.at[0, 'away'] = [data['away']]
    else:
        for match in data:
            matches_dict = dict([(key,val) for key,val in match.items() if key in columns_req_ls])
            matches_df = pd.DataFrame(matches_dict, columns=columns_req_ls).reset_index(drop=True)
    
    matches_df = matches_df.set_index('matchId')        
    return matches_df


def load_EPV_grid(fname=r'/Users/admin/dev/algobetting/infra/data/collectors/whoscored/EPV_grid.csv'):
    epv = np.loadtxt(fname, delimiter=',')
    return epv


def get_EPV_at_location(position, EPV, attack_direction, field_dimen=(106., 68.)):
    x, y = position
    if abs(x) > field_dimen[0]/2. or abs(y) > field_dimen[1]/2.:
        return 0.0
    else:
        if attack_direction == -1:
            EPV = np.fliplr(EPV)
        ny, nx = EPV.shape
        dx = field_dimen[0]/float(nx)
        dy = field_dimen[1]/float(ny)
        ix = (x+field_dimen[0]/2.-0.0001)/dx
        iy = (y+field_dimen[1]/2.-0.0001)/dy
        return EPV[int(iy), int(ix)]


def to_metric_coordinates_from_whoscored(data, field_dimen=(106., 68.)):
    x_columns = [c for c in data.columns if c[-1].lower()=='x'][:2]
    y_columns = [c for c in data.columns if c[-1].lower()=='y'][:2]
    x_columns_mod = [c+'_metrica' for c in x_columns]
    y_columns_mod = [c+'_metrica' for c in y_columns]
    data[x_columns_mod] = (data[x_columns]/100*106)-53
    data[y_columns_mod] = (data[y_columns]/100*68)-34
    return data


def addEpvToDataFrame(data):
    EPV = load_EPV_grid(r'/Users/admin/dev/algobetting/infra/data/collectors/whoscored/EPV_grid.csv')
    data = to_metric_coordinates_from_whoscored(data)

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
    
    data = data.assign(EPV_difference=EPV_difference)
    
    drop_cols = ['x_metrica', 'endX_metrica', 'y_metrica', 'endY_metrica']
    data.drop(drop_cols, axis=1, inplace=True)
    data.rename(columns={'EPV_difference': 'EPV'}, inplace=True)
    
    return data