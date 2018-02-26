import lxml.html as LH
import pandas as pd
import datetime
import urllib3
import warnings
import os
warnings.filterwarnings("ignore")


def scrape_training_data(start_year, teams):
    """
    Scrapes www.worldfootball.net for historical PL fixture results for given teams

    :param start_year: Defines the first season data is wanted for by the first year of that season
    :type start_year: int
    :param teams: Defines the teams to get fixture history data for
    :type teams: list
    """

    # Connection settings
    # http = urllib3.PoolManager()
    default_headers = urllib3.make_headers(proxy_basic_auth='shephej:Kjowwnim35')
    http = urllib3.ProxyManager("https://10.132.100.135:8080/", headers=default_headers)  # 8080

    now = datetime.datetime.now()
    for team in teams:
        season = start_year
        season_number = 0

        while season <= now.year:

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

    out_path = os.path.join('training_data', 'world_football_fixture_history.csv')
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
    file_path = os.path.join('training_data', 'world_football_fixture_history.csv')
    df = pd.read_csv(file_path)

    # Extract premier league opponents
    team_names = df.loc[df['competition'].str.contains('Premier League'), 'opponent'].str.lower()
    team_names = team_names.str.replace(' ', '-').unique()
    team_names = team_names.tolist()

    return team_names


if __name__ == '__main__':

    # Teams to gather data for
    teams = get_prem_team_names()

    # Season to start from (first year of)
    start_year = 1990

    # Scrape and process
    scrape_training_data(start_year, teams)
