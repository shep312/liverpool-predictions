from lxml import etree
import numpy as np
import pandas as pd
from io import StringIO
import urllib3
import os

def scape_training_data():
    """ Scrapes http://www.lfchistory.net/SeasonArchive/Games/ for historical
    PL fixture results for LFC """

    # Unfortunately the website integers are in a strange order
    first_set = np.arange(33,46)
    second_set = np.array([127, 126, 125, 124, 123, 122, 121, 120, 119, 118,
                           117, 116, 115, 104, 46])
    page_indeces = np.concatenate([first_set, second_set])

    # Loop through website reads and create a dataframe
    for zz in range(len(page_indeces) - 1):

        # Print year
        print(zz)

        # Page to read
        target_page = 'http://www.lfchistory.net/SeasonArchive/Games/{}'.format(str(page_indeces[zz]))

        # Read page and save as HTML
        http = urllib3.PoolManager()
        r = http.request('GET', target_page)
        page = r.data.decode('utf-8')

        # Parse the HTML string
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(page), parser)
        root = tree.getroot()

        # Identify data objects I'm interested in
        tds = root.xpath("//td/text()")
        scores = root.xpath("//a/text()")
        scores = scores[1:]

        # Initialise lists to fill
        i = 0
        index = []
        date = []
        opposition = []
        venue = []
        competition = []

        # Loop through html data and organise it
        while i < len(tds):

            # Stop when the index resets and brings in friendlies
            if int(tds[i]) < i / 5:
                break

            index.append(tds[i])
            date.append(tds[i+1])
            opposition.append(tds[i+2])
            venue.append(tds[i+3])
            competition.append(tds[i+4])

            i += 5

        # Extract the scores
        home_score = []
        away_score = []
        position = 0

        for score in scores:
            home_score.append(score[0])
            away_score.append(score[4])

            # If its not a number then the scores have finished
            try:
                x = int(score[0])
            except ValueError:
                break

            position += 1
            if position >= len(index):
                break

        # Create a dataframe of the results
        fixture_history = pd.DataFrame({
            'nth_game_this_season': index,
            'date': date,
            'opposition': opposition,
            'venue': venue,
            'competition': competition,
            'home_score': home_score,
            'away_score': away_score
        })

        # Combine results
        if zz == 0:
            final_df = pd.DataFrame({
                'nth_game_this_season': [],
                'date': [],
                'opposition': [],
                'venue': [],
                'competition': [],
                'home_score': [],
                'away_score': []
            })
        else:
            final_df = pd.concat([final_df, fixture_history])

    # Convert date to a datetime
    final_df['date'] = pd.to_datetime(final_df['date'], format='%d.%m.%Y')

    # Scores as ints
    final_df['away_score'] = final_df['away_score'].astype(np.int32)
    final_df['home_score'] = final_df['home_score'].astype(np.int32)

    out_path = os.path.join(os.pardir, 'data', 'training_data', 'liverpool_fixture_history.csv')
    final_df.to_csv(out_path, index=False, encoding='utf-8')

    return
