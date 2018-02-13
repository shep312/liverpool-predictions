import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from data.data_processing import ProcessInput
from data.data_importation import scrape_training_data
from train.train import train_lgbm_model, get_training_metrics
import lightgbm as lgbm


def main(collect_current_data=True):

    # Update training data if necessary
    if collect_current_data:
        print('Loading fixture data...')
        scrape_training_data()
        print('Fixture data loaded.')

    # Load training data
    file_path = os.path.join('data', 'training_data',
                             'liverpool_fixture_history.csv')
    df = pd.read_csv(file_path)

    # Transform and clean data
    processor = ProcessInput()
    df = processor.fit_transform(df)
    df = processor.drop_features(df)

    # Split data
    test_size = 0.2
    X_train, X_test, y_train, y_test, train_weight = \
        processor.stratified_train_test(df, test_size=test_size)

    # Train and predict
    clf = train_lgbm_model(train, y_train, processor.categoricals, train_weight)
    y_prob = clf.predict(test)
    y_pred = y_prob.argmax(axis=1)

    # Print out metrics
    get_training_metrics(y_test, y_pred)

    # Save model and display performance
    clf.save_model(filename=os.path.join('models',
                                         'lgbm_result_classifier.txt'))
    lgbm.plot_importance(clf)


if __name__ == '__main__':

    #TODO add logging

    main()
