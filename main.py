import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from utilities.data_processing import ProcessInput


def main():

    file_path = os.path.join('data', 'liverpool_fixture_history.csv')
    df = pd.read_csv(file_path)
    df = df[df['competition'] == 'Premier League']

    processor = ProcessInput()
    df = processor.fit_transform(df)

if __name__ == '__main__':

    main()
