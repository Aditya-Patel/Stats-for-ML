"""
Aditya Patel
Implementation of the 10-Classification model on 
Stat 598 - Statistical Machine Learning 
"""

import numpy as np
import torch
from torch import nn

class MNISTModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super.__init__()
        
        # Define NN layer stack - 3 features
        self.layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.Linear(hidden_units, output_features)
        )

        def pass_forward(self, x):
            return self.layer_stack(x)

def MNISTDriver(Xtrain: np.ndarray, yTrain: np.ndarray, XTest:np.ndarray, yTest:np.ndarray, out_features=1, hidden_layers=8, neurons=10, optimizer='sgd'):
    # Check shapes of labels and targets
    print("Shape of training dataset:\n\tX:{}\n\tY:{}".format(Xtrain.shape, yTrain.shape[1]))
    print("Shape of test dataset:\n\tX:{}\n\tY:{}".format(XTest.shape, yTest.shape))

    # Convert to pyTorch Tensors
    X_train_tensor = torch.from_numpy(Xtrain)
    X_test_tensor = torch.from_numpy(XTest)
    y_train_tensor = torch.from_numpy(yTrain)
    y_test_tensor = torch.from_numpy(yTest)

    in_features = Xtrain.shape[1]
    try:
        out_features = yTrain.shape[1]
    except IndexError:
        out_features = 1
    hidden_units = 8

    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model
    mnist_mod = MNISTModel(in_features, out_features, hidden_units).to(device)
    