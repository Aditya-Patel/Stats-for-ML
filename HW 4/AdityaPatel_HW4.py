"""
Aditya Patel
Homework 4
Stat 598 - Statistical Machine Learning 
"""
import pandas as pd
import numpy as np
import os

from MNISTImplementation import MNISTDriver
import GANImplementation as GAN

MNIST_TRAIN_DIR = os.getcwd() + '/mnist_train.csv'
MNIST_TEST_DIR = os.getcwd() + '/mnist_test.csv'

SAMPLE_CT = 1600
COMPONENT_CT = 8
STD_DEV = 0.02
LEARNING_RATE = 0.0001
EPOCH_CT = 5000
BATCH_SZ = 8
INPUT_SZ = 2


def ImportMNISTDataFromDirectory(train_dir, test_dir):
    mnist_test = pd.read_csv(test_dir)
    mnist_train = pd.read_csv(train_dir)

    mnist = pd.concat([mnist_train, mnist_test])
    mnist.columns = mnist_train.columns

    X_df = mnist.drop(['label'], axis=1)
    y_df = mnist.label
    return X_df, y_df

if __name__ == '__main__':
    test, train = ImportMNISTDataFromDirectory(MNIST_TRAIN_DIR, MNIST_TEST_DIR)

    MNISTDriver(test, train)
    GAN.driver(COMPONENT_CT, SAMPLE_CT, STD_DEV, INPUT_SZ, LEARNING_RATE, BATCH_SZ, EPOCH_CT)