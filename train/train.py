import lightgbm as lgbm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def train_lgbm_model(train, y_train, categoricals, train_weight):
    """ Train a LightGBM model on a training set and predict on a test """

    # Create a training set
    categoricals = ['opposition', 'liverpool_at_home', 'day_of_week']
    lgbm_train_set = lgbm.Dataset(data=train, label=y_train,
                                  categorical_feature=categoricals,
                                  weight=train_weight,
                                  free_raw_data=False)

    # Set the training parameters
    lgbm_params = {'application': 'multiclass',
                   'booting': 'gbdt',
                   'metric': 'multi_logloss',
                   'training_metric': True,
                   'learning_rate': 0.05,
                   'feature_fraction': 0.8,
                   'min_data_in_leaf': 30,
                   'num_leaves': 31,
                   'num_classes': 3}

    # Train
    clf = lgbm.train(train_set=lgbm_train_set,
                 params=lgbm_params,
                 num_boost_round=best_iteration)

    return clf

def get_training_metrics(y_test, y_pred):

    # Simple accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    print("LGBM classification accuracy =  \t {:.2f}%".format(100*accuracy))
    print("Precision =  \t " + precision)
    print("Recall =  \t " + recall)

    # Visualise the confusion matrix, normalised for classification frequency
    conf_matrix = confusion_matrix(y_test, y_pred)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_conf_matrix = conf_matrix / row_sums
    # np.fill_diagonal(norm_conf_matrix, 0)

    fig, ax = plt.subplots(figsize=[7,5])
    conf_plot = ax.matshow(norm_conf_matrix, cmap=plt.cm.gray)
    plt.xlabel('Predicted class')
    plt.ylabel('Actual class')
    plt.colorbar(ax=ax, mappable=conf_plot)
    plt.show()

    return
