import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class ProcessInput:
    """ A class that processes raw data for the liverpool model """

    def __init__(self):
        """ Constructor """
        self.opposition_encoder = LabelEncoder()
        self.opposition_count_dict = {}
        self.beatability_df = pd.DataFrame()

    def fit(self, df):
        """ Fits all the objects that require fitting """

        # Fit and save opposition encoder
        df['opposition'] = opposition_encoder.fit_transform(df['opposition'])
        pickle.dump(opposition_encoder, open('models/opposition_encoder.p', 'wb'))


    def transform(self, df):
        """ Transform the data once encoders fitted """

        # TODO define this functionality.

        return df


    def fit_transform(self, df):
        """ Main transformation function """

        # Extract results
        df = self.define_results(df)

        # Convert the date to datetime
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

        # Get some opposition metrics
        df = self.get_beatability_index(df)

        # Fit and save opposition encoder
        self.opposition_encoder = LabelEncoder()
        df['opposition'] = self.opposition_encoder.fit_transform(df['opposition'])
        pickle.dump(self.opposition_encoder, open('models/opposition_encoder.p', 'wb'))

        # Extract extra info from the datetime
        df = self.process_date(df)

        # Get the current streaks
        df = self.calculate_streaks(df)

        # Season data
        df = self.get_season_data(df)

        # Add the number of times the two teams have played each other in this
        # data set
        self.opposition_count_dict = dict(df['opposition'].value_counts())
        df['n_times_teams_played'] = df['opposition'].map(self.opposition_count_dict)

        return df


    def define_results(self, df):
        """ Takes the goals scored and the venue and converts to wins and
        losses """

        # Location. Extract whether it was a home game or not then lose the
        # venue as its unlikely we can learn anything that we can't get from
        # home/away and the opposition
        df['liverpool_at_home'] = df['venue'] == 'Anfield'
        df.drop('venue', axis=1, inplace=True)

        df['result'] = 0

        # Convert home score and away score into liverpool/opposition goals
        df.loc[df['liverpool_at_home'], 'liverpool_goals_scored'] = df.loc[df['liverpool_at_home'], 'home_score']
        df.loc[df['liverpool_at_home'], 'opposition_goals_scored'] = df.loc[df['liverpool_at_home'], 'away_score']

        df.loc[~df['liverpool_at_home'], 'opposition_goals_scored'] = df.loc[~df['liverpool_at_home'], 'home_score']
        df.loc[~df['liverpool_at_home'], 'liverpool_goals_scored'] = df.loc[~df['liverpool_at_home'], 'away_score']

        # Win
        df.loc[df['liverpool_goals_scored'] > df['opposition_goals_scored'], 'result'] = 1

        # Draw
        df.loc[df['liverpool_goals_scored'] == df['opposition_goals_scored'], 'result'] = 0

        # Loss
        df.loc[df['liverpool_goals_scored'] < df['opposition_goals_scored'], 'result'] = 2

        # Win / Not win binary flag
        df['win_flag'] = df['result'] == 1
        df['loss_flag'] = df['result'] == 2

        df.drop(['home_score', 'away_score'], axis=1, inplace=True)

        return df

    def get_beatability_index(self, df):
        """ Define a value between -1 and 1 that indicates how well liverpool
        generally do against this team """

        # Trim data to the last 5 years
        years_to_consider = 5
        time_cutoff = pd.to_datetime('today') - pd.Timedelta(weeks=52 * years_to_consider)
        temp_df = df[df['date'] > time_cutoff]

        # Loop through each team and calculate general performance
        unique_opponents = temp_df['opposition'].unique()
        win_proportion, loss_proportion, times_played = [], [] ,[]

        for team in unique_opponents:
            win_proportion.append(temp_df.loc[temp_df['opposition'] == team, 'win_flag'].mean())
            loss_proportion.append(temp_df.loc[temp_df['opposition'] == team, 'loss_flag'].mean())
            times_played.append(sum(temp_df['opposition'] == team))

        self.beatability_df = pd.DataFrame({'opposition': unique_opponents,
                                            'win_proportion': win_proportion,
                                            'loss_proportion': loss_proportion,
                                            'times_played': times_played})
        self.beatability_df['beatability_index'] = self.beatability_df['win_proportion'] - \
                                                   self.beatability_df['loss_proportion']

        # Merge results back into original df
        df = df.merge(self.beatability_df, left_on='opposition', right_on='opposition', how='left')

        # Any nulls make 0 (neutral)
        df['beatability_index'].fillna(0, inplace=True)

        # Drop columns used in metric calculation. Not crucial but want to keep in streamlined
        df.drop(['times_played', 'win_proportion', 'loss_proportion'], axis=1, inplace=True)

        return df


    def process_date(self, df):
        """ Extracts information from the data such as the week day """

        # Extract info from date
        df['day_of_week'] = df['date'].dt.weekday.astype(int)  # Record day of week

        # Order by date and get days since last game
        df.sort_values(by='date', ascending=False, inplace=True)
        df.reset_index(inplace=True, drop=True)

        for i in range(len(df) - 1):
            df.loc[i, 'days_since_last_game'] = df.loc[i, 'date'] - df.loc[i + 1, 'date']
        df['days_since_last_game'].fillna(df['days_since_last_game'].median(), inplace=True)
        df['days_since_last_game'] = df['days_since_last_game'].dt.days.astype(int)

        # Cap days since last game to 10
        df.loc[df['days_since_last_game'] > 10, 'days_since_last_game'] = 10

        # Convert date into a weighting for the training data
        df['date'] = pd.to_datetime('today') - df['date']  # Convert to difference from today
        df['date'] = df['date'].dt.days.astype(float)  # Convert to an integer

        scaler = MinMaxScaler(feature_range=(0,1))
        df['date'] = scaler.fit_transform(df['date'].values.reshape(-1, 1))

        # Invert so the older values get less weight
        df['date'] = 1 - df['date']

        return df


    def calculate_streaks(self, df):
        """ Use recent results to calculate the current win and undefeated
        streaks"""

        # Reset the index so we can loop down in time
        df.reset_index(drop=True, inplace=True)

        # Initialise a win streak feature
        df['win_streak'] = 0
        df['undefeated_streak'] = 0

        for i in range(len(df) - 1):

            # Initialise streak counters
            win_streak = 0
            undefeated_streak = 0

            # Was the most recent game a win?
            won_last_game = bool(df.loc[i + 1, 'result'] == 1)

            # Not a loss?
            undefeated_last_game = bool((df.loc[i + 1, 'result'] == 1) | (df.loc[i + 1, 'result'] == 0))

            # Keep going back until we break the win streak
            while won_last_game:
                win_streak += 1
                index = i + 1 + win_streak
                if index < len(df):
                    won_last_game = bool(df.loc[index, 'result'] == 1)
                else:
                    break

            # Keep going back until we break the undefeated streak
            while undefeated_last_game:
                undefeated_streak += 1
                index_undef = i + 1 + undefeated_streak
                if index_undef < len(df):
                    undefeated_last_game = bool((df.loc[index_undef, 'result'] == 1) | (df.loc[index_undef, 'result'] == 0))
                else:
                    break

            # Save the result
            df.loc[i, 'win_streak'] = win_streak
            df.loc[i, 'undefeated_streak'] = undefeated_streak

        return df

    def get_season_data(self, df):
        """ Use the game number to calculate some information about the season
        so far """

        # Ensure that only league fixtures are dealt with
        prem_df = df[df['competition'] == 'Premier League']

        # Order in ascending order of time and resent index
        prem_df.sort_values(by='date', ascending=True, inplace=True)
        prem_df.reset_index(drop=True, inplace=True)

        # Initialise features
        prem_df['pl_gameweek'], prem_df['season_number'], prem_df['season_points'] = 0, 0, 0

        # Count the season number
        season_number = 0

        for i in range(len(prem_df) - 1):

            # Check if its a season end
            if i == 0:
                season_end_flag = 1

            elif i > 0 and prem_df.loc[i, 'nth_game_this_season'] < prem_df.loc[i-1, 'nth_game_this_season']:
                season_end_flag = 1

            else:
                season_end_flag = 0

            # Set first value of the season to be 1
            if season_end_flag:
                prem_game_counter = 1
                points_counter = 0
                goals_for_counter = 0
                goals_against_counter = 0
                season_number += 1

            prem_df.loc[i, 'pl_gameweek'] = prem_game_counter
            prem_df.loc[i, 'season_number'] = season_number

            # Increment premier league game
            prem_game_counter += 1

            # Record goal stats
            goals_for_counter += prem_df.loc[i, 'liverpool_goals_scored']
            goals_against_counter += prem_df.loc[i, 'opposition_goals_scored']

            # Calculate points accrued at this stage and therefore points per game (PPG)
            if prem_df.loc[i, 'win_flag']:
                points_counter += 3
            elif ~prem_df.loc[i, 'win_flag'] and ~prem_df.loc[i, 'loss_flag']:
                points_counter += 1

            # Variables must be stored in the next game, otherwise they
            # are unknown at the time of prediction
            prem_df.loc[i + 1, 'PPG'] = points_counter / prem_df.loc[i, 'pl_gameweek']
            prem_df.loc[i + 1, 'season_points'] = points_counter

            # Goals for per game, goals against per game and goal difference
            # per game
            prem_df.loc[i + 1, 'GFPG'] = goals_for_counter / prem_df.loc[i, 'pl_gameweek']
            prem_df.loc[i + 1, 'GAPG'] = goals_against_counter / prem_df.loc[i, 'pl_gameweek']
            prem_df.loc[i + 1, 'GDPG'] = prem_df.loc[i + 1 , 'GFPG'] - prem_df.loc[i + 1, 'GAPG']


        df = df.merge(prem_df[['date', 'pl_gameweek',
                               'PPG', 'season_number',
                               'season_points', 'GFPG',
                               'GAPG', 'GDPG']],
                               left_on='date', right_on='date', how='left')

        return df
