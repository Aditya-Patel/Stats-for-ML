"""
Aditya Patel
Implementation of the 10-Classification model
Stat 598 - Statistical Machine Learning 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

EPOCH_COUNT = 50
BATCH_SIZE = 86

MNIST_TRAIN_DIR = os.getcwd() + '/mnist_train.csv'
MNIST_TEST_DIR = os.getcwd() + '/mnist_test.csv'

def ImportMNISTDataFromDirectory(train_dir, test_dir):
    mnist_test = pd.read_csv(test_dir)
    mnist_train = pd.read_csv(train_dir)

    Xt_df = mnist_test.drop(['label'], axis=1)
    yt_df = mnist_test.label
    Xr_df = mnist_train.drop(['label'], axis=1)
    yr_df = mnist_train.label
    return Xt_df, yt_df, Xr_df, yr_df

def CreateTensorflowDatasets(Xt, yt, Xr, yr):
    X_r = Xr.values.reshape(-1, 28, 28, 1)
    X_t = Xt.values.reshape(-1, 28, 28, 1)
    y_r = tf.keras.utils.to_categorical(yr, num_classes=10)
    y_t = tf.keras.utils.to_categorical(yt, num_classes=10)    

    rescale_data = keras.Sequential([keras.layers.Rescaling(1/255.)])
    X_r, X_v, y_r, y_v = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

    train_ds = tf.data.Dataset.from_tensor_slices((X_r, y_r))
    vald_ds = tf.data.Dataset.from_tensor_slices((X_v, y_v))
    test_ds = tf.data.Dataset.from_tensor_slices((X_t, y_t))

    train_ds = train_ds.map(lambda x, y: (rescale_data(x), y)).shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    vald_ds = vald_ds.map(lambda x, y: (rescale_data(x), y)).shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescale_data(x), y)).shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE) 

    return train_ds, vald_ds, test_ds

if __name__ == '__main__':
    # Create and preprocess test, train, and validationd datasets
    Xt_df, yt_df, Xr_df, yr_df = ImportMNISTDataFromDirectory(MNIST_TRAIN_DIR, MNIST_TEST_DIR)
    train, vald, test = CreateTensorflowDatasets(Xt_df, yt_df, Xr_df, yr_df)

    # Define, compile, and fit NN model
    model = keras.models.Sequential([
        # Image Convolution and downsampling - Node 1
        keras.layers.Conv2D(input_shape=(28,28,1), filters=64, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=2, padding='same'),
        # Image Convolution and downsampling - Node 2
        keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=2, padding='same'),
        # Image Convolution and downsampling - Node 3
        keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=2, padding='same'),
        # Reduce to 1-D vector
        keras.layers.Flatten(),
        # Categorization using DNNs 
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, min_lr=0.0000001)    
    hist = model.fit(train, validation_data = vald, epochs = EPOCH_COUNT, callbacks=[lr_reducer], shuffle=True,)

    # Evaluate against test data
    future = model.evaluate(test, verbose=2)

    # Plot error
    x = np.array(hist.epoch)
    y = np.array(hist.history['accuracy'])
    y = np.multiply(np.subtract(1, y), 100)
    plt.plot(x, y)
    plt.title(f'Test Error in Percent vs. Epoch\nModel Prediction Error = {(1-future[1])*100:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Test Error (%)')
    plt.show()