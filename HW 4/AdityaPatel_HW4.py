"""
Aditya Patel
Homework 4
Stat 598 - Statistical Machine Learning 
"""
import pandas as pd
import numpy as np
import os

import GANImplementation as GAN


NN_EPOCH_COUNT = 50
NN_BATCH_SIZE = 86
MNIST_TRAIN_DIR = os.getcwd() + '/mnist_train.csv'
MNIST_TEST_DIR = os.getcwd() + '/mnist_test.csv'

SAMPLE_CT = 1600
COMPONENT_CT = 8
STD_DEV = 0.02
LEARNING_RATE = 0.0001
EPOCH_CT = 5000
BATCH_SZ = 8
INPUT_SZ = 2


if __name__ == '__main__':
    # MNISTDriver(test, train)
    GAN.driver(COMPONENT_CT, SAMPLE_CT, STD_DEV, INPUT_SZ, LEARNING_RATE, BATCH_SZ, EPOCH_CT)