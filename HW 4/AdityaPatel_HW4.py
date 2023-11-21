"""
Aditya Patel
Homework 4
Stat 598 - Statistical Machine Learning 
"""
import pandas as pd
import numpy as np
import os

from MNISTImplementation import MNISTDriver
from GANImplementation import GAN_Driver

num_samples = 1600
num_comps = 8
std_dev = 0.02
latent_space = 100
epochs = 3000

def CreateMNISTDataset(dir):
    df = pd.read_csv(dir)
    Xdf = df.iloc[:, 1:].to_numpy()
    ydf = df.iloc[:, 0].to_numpy()
    return Xdf, ydf

if __name__ == '__main__':
    # train_cwd = os.getcwd() + '/mnist_train.csv'
    # test_cwd = os.getcwd() + '/mnist_test.csv'

    # X_train, y_train = CreateMNISTDataset(train_cwd)
    # X_test, y_test = CreateMNISTDataset(test_cwd)

    # MNISTDriver(X_train, y_train, X_test, y_test, out_features=10, hidden_layers=5, neurons=10)
    GAN_Driver(num_samples=num_samples, num_comps=num_comps, std_dev=std_dev, latent_space=latent_space, epochs=epochs)