"""
Aditya Patel
Homework 4
Stat 598 - Statistical Machine Learning 
"""
import pandas as pd
import numpy as np
import os

from MNISTImplementation import MNISTDriver
# from GANImplementation import GanDriver

def CreateMNISTDataset(dir):
    df = pd.read_csv(dir)
    Xdf = df.iloc[:, 1:].to_numpy()
    ydf = df.iloc[:, 0].to_numpy()
    return Xdf, ydf

if __name__ == '__main__':
    train_cwd = os.getcwd() + '/mnist_train.csv'
    test_cwd = os.getcwd() + '/mnist_test.csv'

    X_train, y_train = CreateMNISTDataset(train_cwd)
    X_test, y_test = CreateMNISTDataset(test_cwd)

    MNISTDriver(X_train, y_train, X_test, y_test, out_features=10, hidden_layers=5, neurons=10)
    # GanDriver()