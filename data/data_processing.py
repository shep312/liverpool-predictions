import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit


class ProcessInput:
    """ A class that processes raw data for the liverpool model """

    def __init__(self):
        """ Constructor. Initialise some of the variable types and define
        those features that are categorical """

        self.opponent_encoder = LabelEncoder()
        self.opponent_count_dict = {}
        self.beatability_df = pd.DataFrame()
        self.categoricals = ['opponent', 'place', 'day_of_week']

    def fit(self, df):
        """ Fits all the objects that require fitting """

        # Fit and save opponent encoder
        df['opponent'] = self.opponent_encoder.fit_transform(df['opponent'])
        pickle.dump(self.opponent_encoder, open('models/opponent_encoder.p', 'wb'))

    def transform(self, df):
        """ Transform the data once encoders fitted """

        # TODO define this functionality.

        return df

    def fit_transform(self, df):
        """ Main transformation function """

        # Convert the date to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # Get some opponent metrics
        df = self.get_beatability_index(df)

        # Fit and save opponent encoder
        self.opponent_encoder = LabelEncoder()
        df['opponent'] = self.opponent_encoder.fit_transform(df['opponent'])
        pickle.dump(self.opponent_encoder, open('opponent_encoder.p', 'wb'))

        # Extract extra info from the datetime
        df = self.process_date(df)

        # Get the current streaks
        df = self.calculate_streaks(df)

        # Season data
        df = self.get_season_data(df)

        # Add the number of times the two teams have played each other in this
        # data set
        self.opponent_count_dict = dict(df['opponent'].value_counts())
        df['n_times_teams_played'] = df['opponent'].map(self.opponent_count_dict)

        # Only interested in the Premier League for now
        df = df[df['competition'] == 'Premier League']

        return df

    def get_beatability_index(self, df):
        """ Define a value between -1 and 1 that indicates how well the team
        generally do against this opposition """

        # Trim data to the last 5 years
        years_to_consider = 5
        time_cutoff = pd.to_datetime('today') - pd.Timedelta(weeks=52 * years_to_consider)
        temp_df = df[df['date'] > time_cutoff]

        # Group by eacm combination of team/opponent
        grouping_fun = {'team_win': ['mean'],
                        'team_draw': ['mean'],
                        'date': ['count']}
        self.beatability_df = \
            temp_df.groupby(['team', 'opponent'])['team_win', 'team_draw', 'date'].agg(grouping_fun)
        self.beatability_df.reset_index(inplace=True)
        self.beatability_df.columns = ['team', 'opponent', 'win_proportion', 'loss_proportion', 'times_played']
        self.beatability_df['beatability_index'] = self.beatability_df['win_proportion'] - \
                                                   self.beatability_df['loss_proportion']

        # Merge results back into original df
        df = df.merge(self.beatability_df, left_on=['team', 'opponent'], right_on=['team', 'opponent'], how='left')

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
        df.sort_values(by='date', ascending=True, inplace=True)
        df.reset_index(inplace=True, drop=True)
        for team in df['team'].unique():
            counter = 0
            for i in range(len(df)):
                if df.loc[i, 'team'] == team:
                    if counter == 0:
                        df.loc[i, 'days_since_last_game'] = pd.Timedelta('nan')
                        prev_time = df.loc[i, 'date']
                    else:
                        curr_time = df.loc[i, 'date']
                        df.loc[i, 'days_since_last_game'] = curr_time - prev_time
                        prev_time = curr_time
                    counter += 1
        df.sort_values(by='date', ascending=False, inplace=True)

        df['days_since_last_game'] = pd.to_timedelta(df['days_since_last_game'])
        df['days_since_last_game'].fillna(df['days_since_last_game'].max(), inplace=True)
        df['days_since_last_game'] = df['days_since_last_game'].dt.days.astype(int)

        # Cap days since last game to 10
        df.loc[df['days_since_last_game'] > 10, 'days_since_last_game'] = 10

        # Convert date into a weighting for the training data
        df['date'] = pd.to_datetime('today') - df['date']  # Convert to difference from today
        df['date'] = df['date'].dt.days.astype(float)  # Convert to an integer

        scaler = MinMaxScaler(feature_range=(0, 1))
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
            undefeated_last_game = bool(df.loc[i + 1, 'result'] != 2)

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
                    undefeated_last_game = bool(df.loc[index_undef, 'result'] != 2)
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

            elif i > 0 and prem_df.loc[i, 'nth_game_this_season'] < prem_df.loc[i - 1, 'nth_game_this_season']:
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
            goals_for_counter += prem_df.loc[i, 'liverpool_score']
            goals_against_counter += prem_df.loc[i, 'opponent_score']

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
            prem_df.loc[i + 1, 'GDPG'] = prem_df.loc[i + 1, 'GFPG'] - prem_df.loc[i + 1, 'GAPG']

        # Final row is the most recent game. If the last game wasn't the final
        # one in a season then increment the gameweek
        if prem_df.loc[i, 'pl_gameweek'] < 38:
            prem_df.loc[i + 1, 'pl_gameweek'] = prem_df.loc[i, 'pl_gameweek'] + 1
        else:
            prem_df.loc[i + 1, 'pl_gameweek'] = 1

        df = df.merge(prem_df[['date', 'pl_gameweek',
                               'PPG', 'season_number',
                               'season_points', 'GFPG',
                               'GAPG', 'GDPG']],
                      left_on='date', right_on='date', how='left')

        return df

    def drop_features(self, df):
        """ Drop features that are either leaks of the label or aren't desired
        for training. Includes:
            - competition: No variation, only PL
            - liverpool_score/opponent_score: Not known at
              inference
            - season_number: Not thought to be useful
            - season_points: Not thought to be useful
            - win_flag/loss_flag: Gives the label away
        """

        df.drop(['competition',
                 'liverpool_score',
                 'opponent_score',
                 'season_number',
                 'season_points',
                 'win_flag',
                 'loss_flag'], axis=1, inplace=True)

        return df

    def stratified_train_test(self, df, test_size=0.1):
        """ Split input data into train and test sets. Since the dataset is
        small, stratify by label to make sure there the sets are representative
        of each other at least in the label."""

        # Make splitter
        split = StratifiedShuffleSplit(n_splits=1,
                                       test_size=test_size,
                                       random_state=42)

        # Get startified indices and split
        df.reset_index(inplace=True, drop=True)
        for train_index, test_index in split.split(df, df['result']):
            train = df.loc[train_index]
            test = df.loc[test_index]

        # Check and print distributions
        sample_comparison = pd.DataFrame({
            'overall': df['result'].value_counts().sort_index() / len(df),
            'stratified': test['result'].value_counts().sort_index() / len(test),
        })
        sample_comparison['strat_perc_error'] = \
            100 * (sample_comparison['stratified'] - sample_comparison['overall']) \
            / sample_comparison['overall']

        # TODO and print here

        # Separate the labels
        y_train = train.pop('result')
        y_test = test.pop('result')

        # Drop dates, but keep the training dates as a weight
        train_weight = train.pop('date')
        test.drop('date', axis=1, inplace=True)

        return train, test, y_train, y_test, train_weight


def main():
    file_path = os.path.join('training_data', 'world_football_fixture_history.csv')
    df = pd.read_csv(file_path)
    processor = ProcessInput()
    df = processor.fit_transform(df)

    return


if __name__ == '__main__':
    main()
