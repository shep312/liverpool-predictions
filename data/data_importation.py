import lxml.html as LH
import pandas as pd
import datetime
import urllib3
import os


def scrape_training_data(start_year):
    """ Scrapes http://www.lfchistory.net/SeasonArchive/Games/ for historical
    PL fixture results for LFC """

    # Teams to gather data for
    teams = ['liverpool-fc']

    # Connection settings
    # http = urllib3.PoolManager()
    default_headers = urllib3.make_headers(proxy_basic_auth='shephej:Kjowwnim35')
    http = urllib3.ProxyManager("https://10.132.100.135:8080/", headers=default_headers)  # 8080

    season = start_year
    season_number = 0
    now = datetime.datetime.now()
    for team in teams:

        while season <= now.year:

            # Page to read
            target_page = 'https://www.worldfootball.net/teams/{}/{}/3/'.format(team, season)

            # Read page to get HTML
            r = http.request('GET', target_page)
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
                    table_data.append(row)

            header.append('competition')
            header.append('season')
            header.append('season_number')

            # Create data frame
            if 'df' in locals():
                df = df.append(pd.DataFrame(table_data, columns=header), ignore_index=True)
            else:
                df = pd.DataFrame(table_data, columns=header)

            # Increment year
            print('Data acquired for team {}, season {} / {}'.format(team, season - 1, season))
            season += 1
            season_number += 1

    # Drop empty columns resulting from the HTML parse
    df.drop('', axis=1, inplace=True)

    # Convert the date to a datetime and get a round number by comp
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['date_as_int'] = pd.to_numeric(df['date'])
    grouped = df.groupby('competition')
    df['round_number'] = grouped['date_as_int'].rank(method='min')
    df.drop('date_as_int', axis=1, inplace=True)

    # Extract the goals scored by each team
    df['team_score'] = df['result'].str[0]
    df['opposition_score'] = df['result'].str[2]
    df['team_ht_score'] = df['result'].str[5]
    df['opposition_ht_score'] = df['result'].str[7]

    # Results
    df['team_win'] = df['team_score'] > df['opposition_score']
    df['team_draw'] = df['team_score'] == df['opposition_score']
    df['team_loss'] = df['team_score'] < df['opposition_score']

    out_path = os.path.join('data', 'training_data', 'liverpool_fixture_history.csv')
    df.to_csv(out_path, index=False, encoding='utf-8')

    return


def text(elt):
    """Process HTML objects to readable strings"""
    return elt.text_content().replace(u'\xa0', u' ').replace(u'\t', u'').replace(u'\r', u'').replace(u'\n', u'')