'''
Name: Aditya Patel
PUID: PATE1854
STAT 598 - Assignment 2
Summary: Implement Single Tree, Bagging, Random Forest, and Boosting classifiers to identify digits based on images
'''

import numpy as np
import pandas as pd
import os

from numba import jit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

test_data = '/archive/mnist_test.csv'
train_data = '/archive/mnist_train.csv'

st_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 100, 1000],
    'max_features': ['sqrt', 'log2']
}

rf_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [10, 100, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 250, 500]
}

bg_params = {
    'n_estimators': [100, 250, 500],
    'max_samples': [1, 2, 5],
    'max_features': [1, 10, 100],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
}

bst_params = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 250, 500],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.1, 0.5, 1],
    'max_features': ['sqrt', 'log2']
}


def CreateDataset(dir):
    df = pd.read_csv(dir)
    df = df[(df['label'] == 5) | (df['label'] == 8)]
    Xdf = df.iloc[:, 1:]
    ydf = df.iloc[:, 0]
    return Xdf, ydf

def GetBestParams(model, params, trainset_X, trainset_y):
    print("Searching parameter grid for {}".format(model))
    grid_search = GridSearchCV(
        estimator=model, param_grid=params, scoring='accuracy', n_jobs=-1, cv=2, verbose=2)
    grid_search.fit(trainset_X, trainset_y)
    print("Found best params for {}:\n{}".format(model, grid_search.best_params_))
    return grid_search.best_params_


if __name__ == '__main__':
    train_data_path = os.getcwd() + train_data
    test_data_path = os.getcwd() + train_data

    X_train, y_train = CreateDataset(train_data_path)
    X_test, y_test = CreateDataset(test_data_path)
    print("Created test and train datasets")
    
    single_tree = DecisionTreeClassifier()
    random_forest = RandomForestClassifier()
    bagging = BaggingClassifier()
    boosting = GradientBoostingClassifier()
    print("Developed models")

    single_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    bagging.fit(X_train, y_train)
    boosting.fit(X_train, y_train)
    print("Fit models to training data")

    models = [single_tree, random_forest, bagging, boosting]
    params = [st_params, rf_params, bg_params, bst_params]

    print("Getting best parameters")
    best_params = []
    for model, param_set in zip(models, params):
        best_params.append(GetBestParams(model, param_set, X_train, y_train))

    print("Best Identified Parameters")
    for par_set in best_params:
        print(par_set)

    # {'criterion': 'log_loss', 'max_depth': 100, 'max_features': 'sqrt', 'splitter': 'best'}
    # {'criterion': 'gini', 'max_depth': 100, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 250}
    # {'bootstrap': False, 'bootstrap_features': False, 'max_features': 100, 'max_samples': 5, 'n_estimators': 500}
    # {'learning_rate': 0.2, 'loss': 'exponential', 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 500, 'subsample': 1}

    print("Retraining models with best parameters")
    single_tree = DecisionTreeClassifier(**best_params[0])
    random_forest = RandomForestClassifier(**best_params[1])
    bagging = BaggingClassifier(**best_params[2])
    boosting = GradientBoostingClassifier(**best_params[3])

    print("Refit models to training data")
    single_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    bagging.fit(X_train, y_train)
    boosting.fit(X_train, y_train)

    print("Predictions on test dataset")
    st_preds = single_tree.predict(X_test)
    rf_preds = random_forest.predict(X_test)
    bg_preds = bagging.predict(X_test)
    bst_preds = boosting.predict(X_test)

    # Misclassification rate = 1 - accuracy
    accs = [accuracy_score(y_test, st_preds), accuracy_score(y_test, rf_preds), accuracy_score(y_test, bg_preds), accuracy_score(y_test, bst_preds)]
    
    miss = []
    for score in accs:
        miss.append(1-score)

    print(
        "Misclassification report:\n\tSingle Tree: {}\n\tRandom Forest: {}\n\tBagging Classifier: {}\n\tGradient Boosting Classifier: {}".format(miss[0], miss[1], miss[2], miss[3])
    )