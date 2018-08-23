import lxml.html as LH
import pandas as pd
import numpy as np
from sklearn.utils.extmath import cartesian
import json
import datetime
import urllib3
import warnings
import os
warnings.filterwarnings("ignore")


def scrape_training_data(start_year, out_path):
    """
    Scrapes www.worldfootball.net for historical PL fixture results for given teams

    :param start_year: Defines the first season data is wanted for by the first year of that season
    :type start_year: int
    :param out_path: Specifies path to save data to
    :type teams: string
    """

    # Teams to gather data for
    teams = get_prem_team_names()

    # Connection settings
    http = urllib3.PoolManager()

    # Time to collect data for. Worlfootball.net refers to seasons by their FINAL year
    now = datetime.datetime.now()

    # Get the end season. If it is after August, this is next year
    if now.month >= 8:
        end_season = now.year + 1
    else:
        end_season = now.year
    print('Importing data from season {}/{} to season {}/{}'.\
        format(start_year-1, start_year, end_season-1, end_season))

    # Loop through teams to gather the data
    for team in teams:
        season = start_year
        season_number = season - 1990

        while season <= end_season:

            # Page to read
            target_page = 'https://www.worldfootball.net/teams/{}/{}/3/'.format(team, season)

            # Read page to get HTML
            try:
                r = http.request('GET', target_page)
            except:
                print('Couldn\'t find data for team {}'.format(team))
                break
            page = r.data.decode('utf-8')

            # Parse HTML
            root = LH.fromstring(page)
            table = root.xpath('//table')[0]
            header = ['round', 'date', 'time', 'place', '', 'opponent', 'result', '']
            data = [[text(td) for td in tr.xpath('td')]
                    for tr in table.xpath('//tr')]
            current_competition = None
            table_data = []
            for row in data:

                if len(row) == 1:
                    current_competition = row[0]

                if current_competition is not None and len(row) == len(header):
                    row.append(current_competition)
                    row.append('{} / {}'.format(season - 1, season))
                    row.append(season_number)
                    row.append(team)
                    table_data.append(row)

            header.append('competition')
            header.append('season')
            header.append('season_number')
            header.append('team')

            # Create data frame
            if 'df' in locals():
                df = df.append(pd.DataFrame(table_data, columns=header), ignore_index=True)
            else:
                df = pd.DataFrame(table_data, columns=header)

            # Increment year
            print('Data acquired for team {}, season {} / {}'.format(team, season - 1, season))
            season += 1
            season_number += 1

    # TODO check why time isn't coming through. Drop in the meantime
    df.drop('time', axis=1, inplace=True)

    # Drop empty columns resulting from the HTML parse
    df.drop('', axis=1, inplace=True)

    # Convert the date to a datetime and get a round number by comp
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['date_as_int'] = pd.to_numeric(df['date'])
    grouped = df.groupby(['team', 'competition'])
    df['round_number'] = grouped['date_as_int'].rank(method='min')
    df.drop('date_as_int', axis=1, inplace=True)

    # Extract the goals scored by each team
    df['team_score'] = df['result'].str[0]
    df['opposition_score'] = df['result'].str[2]
    df['team_ht_score'] = df['result'].str[5]
    df['opposition_ht_score'] = df['result'].str[7]

    # Convert home or away to binary
    df['team_at_home'] = 0
    df.loc[df['place'] == 'H', 'team_at_home'] = 1
    df.drop('place', axis=1, inplace=True)

    # Results
    df['team_win'] = df['team_score'] > df['opposition_score']
    df['team_draw'] = df['team_score'] == df['opposition_score']
    df['team_loss'] = df['team_score'] < df['opposition_score']

    # Write to a csv
    if os.path.exists(out_path):
        pd.read_csv(out_path).append(df).drop_duplicates().to_csv(out_path, index=False, encoding='utf-8')
    else:
        df.to_csv(out_path, index=False, encoding='utf-8')

    return


def text(elt):
    """Process HTML objects to readable strings"""
    return elt.text_content().replace(u'\xa0', u' ').replace(u'\t', u'').replace(u'\r', u'').replace(u'\n', u'')


def get_prem_team_names():
    """
    Reads a season of liverpool's history and assumes all the teams played are those we're interested in.
    Compiles the team names and returns as a list
    :return: teams_names: list of team names as strings
    """

    # Load .csv
    file_path = os.path.join('data', 'training_data', 'team_names.csv')
    df = pd.read_csv(file_path)

    # Extract premier league opponents
    team_names = df['opponent'].str.lower()
    team_names = team_names.str.replace(' ', '-').unique()
    team_names = team_names.tolist()

    return team_names


def get_stadium_distances():
    """
    Takes a list of UK teams and finds the travel distances between them all. Saves results to .csv
    :return:
    """

    # API call parameters
    api_key = 'AIzaSyAeqR4U6zS_6W7_IcpsQzvXZijuvlT9oGU'
    http = urllib3.PoolManager()

    # Go through each combo and get stadium distance
    unique_teams = np.array(get_prem_team_names())
    lat, lon = [], []
    for i in range(len(unique_teams)):

        # Update user
        if i % 10 == 0:
            print('Getting geolocation data for team: {:d} / {:d}'.format(i, len(unique_teams)))

        # Get latitude and longtitude of team a ('home')
        home_address = unique_teams[i].replace('-', '+')
        latlong_api = 'https://maps.googleapis.com/maps/api/geocode/json?address={},uk&type=stadium&key={}' \
            .format(home_address, api_key)

        r = http.request('GET', latlong_api)
        latlong_data = json.loads(r.data.decode('utf-8'))['results'][0]

        lat.append(latlong_data['geometry']['location']['lat'])
        lon.append(latlong_data['geometry']['location']['lng'])

    # Store location in a team df
    team_locations = pd.DataFrame({'team': unique_teams, 'latitude': lat, 'longitude': lon})

    # All combinations of stadia
    team_combos = cartesian((unique_teams, unique_teams))
    duplicate_mask = team_combos[:, 0] == team_combos[:, 1]
    team_combos = pd.DataFrame(team_combos[~duplicate_mask, :], columns=['team_a', 'team_b'])

    # Append locations
    team_combos = team_combos.merge(team_locations, how='left', left_on='team_a', right_on='team', suffixes=('', '_a'))
    team_combos.drop('team', axis=1, inplace=True)
    team_combos = team_combos.merge(team_locations, how='left', left_on='team_b', right_on='team',
                                    suffixes=('_a', '_b'))
    team_combos.drop('team', axis=1, inplace=True)

    # Get distances between teams
    for i in range(len(team_combos)):

        # Update user
        if i % 100 == 0:
            print('Getting distance data for team combo: {:d} / {:d}'.format(i, len(team_combos)))

        distance_api = 'https://maps.googleapis.com/maps/api/distancematrix/json?' \
                       'units=imperial&origins={:.5f},{:.5f}&destinations={:.5f},{:.5f}&key={}' \
            .format(team_combos.loc[i, 'latitude_a'],
                    team_combos.loc[i, 'longitude_a'],
                    team_combos.loc[i, 'latitude_b'],
                    team_combos.loc[i, 'longitude_b'],
                    api_key)

        r2 = http.request('GET', distance_api)
        if json.loads(r2.data.decode('utf-8'))['rows'][0]['elements'][0]['status'] != 'NOT_FOUND':
            distance_data = json.loads(r2.data.decode('utf-8'))['rows'][0]

            # Save the distances and durations
            team_combos.loc[i, 'travel_distance'] = distance_data['elements'][0]['distance']['value']
            team_combos.loc[i, 'travel_duration'] = distance_data['elements'][0]['duration']['value']

        else:
            print('Couldnt find {} to {}'.format(team_combos.loc[i, 'team_a'], team_combos.loc[i, 'team_b']))
            continue

    team_combos.to_csv(os.path.join('data', 'training_data', 'inter_stadium_travel.csv'), index=False)

    return


if __name__ == '__main__':

    # Scrape and process
    start_year = 1990
    file_path = os.path.join('data', 'training_data', 'world_football_fixture_history.csv')
    scrape_training_data(start_year, fie_path)

    # Update travel between stadium distance if there are any new teams
    update_travel_data = False
    if update_travel_data:
        get_stadium_distances()
