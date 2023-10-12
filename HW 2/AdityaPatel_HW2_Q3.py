'''
Name: Aditya Patel
PUID: PATE1854
STAT 598 - Assignment 2
Summary: Implement Single Tree, Bagging, Random Forest, and Boosting classifiers to identify digits based on images
'''

import numpy as np
import pandas as pd
import os
import sys
import random
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

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
    print("Found best params for {}".format(model))
    return grid_search.best_params_


if __name__ == '__main__':
    single_tree = DecisionTreeClassifier()
    random_forest = RandomForestClassifier()
    bagging = BaggingClassifier()
    boosting = GradientBoostingClassifier()
    print("Developed models")

    models = [single_tree, random_forest, bagging, boosting]
    params = [st_params, rf_params, bg_params, bst_params]

    train_data_path = os.getcwd() + train_data
    test_data_path = os.getcwd() + train_data

    X_train, y_train = CreateDataset(train_data_path)
    X_test, y_test = CreateDataset(test_data_path)
    print("Created test and train datasets")

    print("Getting best parameters")
    best_params = []
    for model, param_set in zip(models, params):
        best_params.append(GetBestParams(model, param_set, X_train, y_train))

    print("Best Identified Parameters")
    for par_set in best_params:
        print(par_set)

    print("Retraining models with best parameters")
    single_tree = DecisionTreeClassifier(**best_params[0])
    random_forest = RandomForestClassifier(**best_params[1])
    bagging = BaggingClassifier(**best_params[2])
    boosting = GradientBoostingClassifier(**best_params[3])

    print("Predicting on test set")

    print("")
