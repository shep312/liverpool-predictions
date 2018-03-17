import numpy as np
import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
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

        # Convert the date to datetime. Drop future games
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # Process fixture result
        df = self.process_result(df)

        # Get some opponent metrics
        df = self.get_beatability_index(df)

        # Append travel distance/time between to opponents
        df = self.append_travel_data(df)

        # Fit and save opponent encoder
        self.opponent_encoder = LabelEncoder()
        df['opponent'] = self.opponent_encoder.fit_transform(df['opponent'])
        pickle.dump(self.opponent_encoder, open(os.path.join('models', 'opponent_encoder.p'), 'wb'))

        # Extract extra info from the datetime
        df = self.process_date(df)

        # Get the current streaks
        df = self.calculate_streaks(df)

        # Season data
        df = self.get_season_data(df)

        # Get opponents form
        df = self.calculate_opponents_form(df)

        return df

    def process_result(self, df):
        """
        Process the result feature of the column into an integer. Integers are mapped as:
            0: Draw
            1: Win
            2: Loss
            3: Future fixture

        :param df: Input dataframe of fixtures
        :return: df: Processed output
        """

        now = datetime.now() - timedelta(days=1)
        df.loc[df['team_draw'], 'result'] = 0
        df.loc[df['team_win'], 'result'] = 1
        df.loc[df['team_loss'], 'result'] = 2
        df.loc[df['date'] >= now, 'result'] = 3
        df['result'] = df['result'].astype(int)

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
        self.beatability_df.columns = ['team', 'opponent', 'win_proportion', 'loss_proportion', 'n_times_teams_played']
        self.beatability_df['beatability_index'] = self.beatability_df['win_proportion'] - \
                                                   self.beatability_df['loss_proportion']

        # Merge results back into original df
        df = df.merge(self.beatability_df, left_on=['team', 'opponent'], right_on=['team', 'opponent'], how='left')

        # Any nulls make 0 (neutral)
        df['beatability_index'].fillna(0, inplace=True)

        # Drop columns used in metric calculation. Not crucial but want to keep in streamlined
        df.drop(['win_proportion', 'loss_proportion'], axis=1, inplace=True)

        return df

    def append_travel_data(self, df):
        """
        Append travel data between stadia extracted from Google Maps' API
        :param df: Main dataframe
        :return: df: Dataframe with travel appended
        """

        # Load distances data
        travel_df = pd.read_csv(os.path.join('data', 'training_data', 'inter_stadium_travel.csv'))

        # Rewrite opponent in same format as team
        df['temp_opponent'] = df['opponent'].str.lower().str.replace(' ', '-')

        # Merge in the travel data
        df = df.merge(travel_df[['team_a', 'team_b', 'travel_distance', 'travel_duration']],
                      how='left',
                      left_on=['team', 'temp_opponent'],
                      right_on=['team_a', 'team_b'])

        # Drop the temporary opponent feature
        df.drop(['temp_opponent', 'team_a', 'team_b'], axis=1, inplace=True)

        return df

    def process_date(self, df):
        """ Extracts information from the data such as the week day """

        # Extract info from date
        df['day_of_week'] = df['date'].dt.weekday.astype(int)  # Record day of week

        # Order by date and get days since last game
        df.sort_values(by='date', ascending=True, inplace=True)
        df.reset_index(inplace=True, drop=True)
        df['days_since_last_game'] = df.groupby(['team'])['date'].diff()
        df.sort_values(by='date', ascending=False, inplace=True)

        df['days_since_last_game'].fillna(df['days_since_last_game'].median(), inplace=True)
        df['days_since_last_game'] = df['days_since_last_game'].dt.days.astype(int)

        # Cap days since last game to 14
        df.loc[df['days_since_last_game'] > 14, 'days_since_last_game'] = 14

        # Convert date into a weighting for the training data
        df['date'] = pd.to_datetime('today') - df['date']  # Convert to difference from today
        df['date'] = df['date'].dt.days.astype(float)  # Convert to an integer

        scaler = MinMaxScaler(feature_range=(0, 1))
        df['date'] = scaler.fit_transform(df['date'].values.reshape(-1, 1))

        # Invert so the older values get less weight
        df['date'] = 1 - df['date']

        return df

    def calculate_streaks(self, df):
        """
        Use recent results to calculate the current win and undefeated
        streaks
        """

        # Reset the index and make time ascending so we're summing backwards
        df.sort_values(by='date', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        def grouping_streak_fun(df):
            """Function to groupby by with"""

            # Win streaks
            df['total_win_cumsum'] = (df['team_win'] == True).cumsum()
            df['wins_up_to_nonwin'] = np.nan
            df.loc[df['team_win'] == False, 'wins_up_to_nonwin'] = df['total_win_cumsum']
            df['wins_up_to_nonwin'].fillna(method='ffill', inplace=True)
            df['wins_up_to_nonwin'].fillna(0, inplace=True)
            df['win_streak'] = df['total_win_cumsum'] - df['wins_up_to_nonwin']
            df['win_streak'] = df['win_streak'].shift(1)

            # Undefeated streaks
            df['total_undef_cumsum'] = (df['team_loss'] == False).cumsum()
            df['undef_up_to_loss'] = np.nan
            df.loc[df['team_loss'] == True, 'undef_up_to_loss'] = df['total_undef_cumsum']
            df['undef_up_to_loss'].fillna(method='ffill', inplace=True)
            df['undef_up_to_loss'].fillna(0, inplace=True)
            df['undefeated_streak'] = df['total_undef_cumsum'] - df['undef_up_to_loss']
            df['undefeated_streak'] = df['undefeated_streak'].shift(1)

            # Clean up temporary columns
            df.drop(['total_win_cumsum', 'wins_up_to_nonwin', 'total_undef_cumsum', 'undef_up_to_loss'],
                    axis=1, inplace=True)

            return df

        # Apply the function per team
        df = df.groupby('team').apply(grouping_streak_fun)

        # Reorder back in terms of descending time
        df.sort_values(by='date', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def get_season_data(self, df):
        """ Use the game number to calculate some information about the season
        so far """

        # Clean up scores
        df['team_score'] = df['team_score'].replace('-', '0').replace(' ', '0').astype(int)
        df['opposition_score'] = df['opposition_score'].replace('-', '0')\
            .replace(' ', '0').replace('e', '0').replace(':', '0') \
            .replace('n', '0').astype(int)

        # Ensure that only league fixtures are dealt with
        prem_df = df[df['competition'].str.contains('Premier League')]

        # Order in ascending order of time and resent index
        prem_df.sort_values(by='date', ascending=True, inplace=True)
        prem_df.reset_index(drop=True, inplace=True)

        # Build some season related features. Offset them by -1 to avoid leak of results
        def shifted_cumsum(df):
            """ Offset the cumulative sum by one back in time"""
            df = df.cumsum()
            df = df.shift(1)
            return df

        def shifted_rolling(df):
            """ Offset a rolling sum to get n wins in last 5"""
            df = df.rolling(min_periods=1, window=5).sum()
            df = df.shift(1)
            return df

        prem_df.loc[:, 'season_league_wins'] = prem_df.groupby(['team', 'season'])['team_win'].apply(shifted_cumsum)
        prem_df.loc[:, 'season_league_draws'] = prem_df.groupby(['team', 'season'])['team_draw'].apply(shifted_cumsum)
        prem_df.loc[:, 'season_goals_for'] = prem_df.groupby(['team', 'season'])['team_score'].apply(shifted_cumsum)
        prem_df.loc[:, 'season_goals_against'] = \
            prem_df.groupby(['team', 'season'])['opposition_score'].apply(shifted_cumsum)
        prem_df.loc[:, 'season_points'] = 3 * prem_df['season_league_wins'] + 1 * prem_df['season_league_draws']

        prem_df.loc[:, 'n_recent_wins'] = prem_df.groupby(['team', 'season'])['team_win'].apply(shifted_rolling)
        prem_df.loc[:, 'n_recent_losses'] = prem_df.groupby(['team', 'season'])['team_loss'].apply(shifted_rolling)
        prem_df.loc[:, 'recent_form'] = prem_df['n_recent_wins'] - prem_df['n_recent_losses']

        # Time dependent metrics: Points per game (PPG), goals-for per game (GFPG), goals-against per game (GAPG),
        # goal difference per game (GDPG).
        prem_df.loc[:, 'PPG'] = prem_df['season_points'] / prem_df['round_number']
        prem_df.loc[:, 'GFPG'] = prem_df['season_goals_for'] / prem_df['round_number']
        prem_df.loc[:, 'GAPG'] = prem_df['season_goals_against'] / prem_df['round_number']
        prem_df.loc[:, 'GDPG'] = prem_df['GFPG'] - prem_df['GAPG']

        # Merge the league data back into the overall set
        df = df.merge(prem_df[['team', 'date', 'PPG',
                               'season_points', 'GFPG',
                               'GAPG', 'GDPG', 'recent_form']],
                      left_on=['team', 'date'], right_on=['team', 'date'], how='left')

        # Drop unformated round number
        df.drop('round', axis=1, inplace=True)

        return df

    def calculate_opponents_form(self, df):
        """
        Get the current form of the team's opponent
        :param df: Main dataframe
        :return: df: Main dataframe with opponent features included
        """

        # Create a PL table-style DF to get each teams form per round
        prem_df = df[df['competition'].str.contains('Premier League')]
        table_df = prem_df[['season_number', 'round_number', 'team', 'season_points', 'PPG', 'GDPG', 'recent_form']] \
                   .sort_values(by=['season_number', 'round_number', 'season_points'])
        team_strings = table_df['team'].str.title()

        # Rename the team column to opponent to allow the merge
        table_df['opponent'] = self.opponent_encoder.transform(team_strings.str.replace('Fc', 'FC')
                                                               .str.replace('-', ' ')
                                                               .str.replace('Afc', 'AFC'))
        table_df.drop('team', axis=1, inplace=True)

        # Merge the league data back into the overall set
        df = df.merge(table_df,
                      left_on=['opponent', 'season_number', 'round_number'],
                      right_on=['opponent', 'season_number', 'round_number'],
                      how='left',
                      suffixes=['', '_opponent'])

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
                 'team_score',
                 'opposition_score',
                 'team_ht_score',
                 'opposition_ht_score',
                 'season',
                 'season_number',
                 'season_points',
                 'team_win',
                 'team_draw',
                 'team_loss'], axis=1, inplace=True)

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

    out_path = os.path.join('training_data', 'processed_fixture_history.csv')
    df.to_csv(out_path, index=False)

    return


if __name__ == '__main__':
    main()
