"""
Aditya Patel
Homework 3 - Q2: K-Means Clustering
Stat 598 - Statistical Machine Learning 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

test_data = '/mnist_test.csv'
train_data = '/mnist_train.csv'

def CreateDataset(dir):
    df = pd.read_csv(dir)
    df = df[(df['label'] == 0) | (df['label'] == 5) | (df['label'] == 8) | (df['label'] == 9)]
    Xdf = df.iloc[:, 1:]
    ydf = df.iloc[:, 0]
    return Xdf, ydf

def CreateMergedXY(x_data, y_data):
    mergeX = pd.DataFrame(np.concatenate(x_data))
    mergeY = pd.DataFrame(np.concatenate(y_data))
    return mergeX, mergeY

def KMeansClusteringAccuracy(x_data, y_data):
    #Encode labels to clusters
    encoder = LabelEncoder()
    true_cluster = encoder.fit_transform(np.ravel(y_data))

    #Create model and fit
    kmc_mod = KMeans(n_clusters = 4, init='k-means++', algorithm='lloyd', n_init=10, max_iter=500).fit(x_data)
    preds = kmc_mod.labels_

    # Create output data frame
    kmcdf = pd.DataFrame(X_data)
    kmcdf['predicted_cluster'] = preds
    kmcdf['true_cluster'] = true_cluster
    kmcdf['true_labels'] = encoder.inverse_transform(true_cluster)

    # Plot cluster centers
    fig = plt.figure(figsize=(8, 8))
    num_rows = 1
    num_cols = 4
        
    for i in range(1, len(kmc_mod.cluster_centers_)+1):
        c = kmc_mod.cluster_centers_[i-1]
        ax = fig.add_subplot(num_rows,num_cols,i)    
        ax.axis('off')
        ax.imshow(c.reshape(28, 28), 
                interpolation='none', cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
    plt.show()

    #Calculate centroid accuracy
    clusters = [kmcdf[kmcdf['predicted_cluster'] == 0], kmcdf[kmcdf['predicted_cluster'] == 1], kmcdf[kmcdf['predicted_cluster'] == 2], kmcdf[kmcdf['predicted_cluster'] == 3]]
    majority_labels = [max(clusters[0]['true_cluster'].value_counts()), max(clusters[1]['true_cluster'].value_counts()), max(clusters[2]['true_cluster'].value_counts()), max(clusters[3]['true_cluster'].value_counts())]
    total_labels = [sum(clusters[0]['true_cluster'].value_counts()), sum(clusters[1]['true_cluster'].value_counts()), sum(clusters[2]['true_cluster'].value_counts()), sum(clusters[3]['true_cluster'].value_counts())]

    total_accuracy = sum(majority_labels)/sum(total_labels)
    cluster_accuracy = []
    for i, j in zip(majority_labels, total_labels):
        cluster_accuracy.append(i/j)

    print("""Accuracy Report:
  Total Accuracy:\t{:.4f}
  Cluster 1 Accuracy:\t{:.4f}
  Cluster 2 Accuracy:\t{:.4f}
  Cluster 3 Accuracy:\t{:.4f}
  Cluster 4 Accuracy:\t{:.4f}""".format(total_accuracy, cluster_accuracy[0], cluster_accuracy[1], cluster_accuracy[2], cluster_accuracy[3]))

def IdentifyBestK(x_data, kStart=1, kStop=11, kStep=1):
    inertia = []
    kVal = []

    for k in range(kStart, kStop, kStep):
        print("Iteration {} of {}".format(k, kStop-1))
        kmc_mod = KMeans(n_clusters=k).fit(x_data)
        kVal.append(k)
        inertia.append(kmc_mod.inertia_)

    fig = plt.figure(figsize=(8,8))
    plt.scatter(kVal, inertia)
    plt.plot(kVal, inertia)
    plt.show()

if __name__ == '__main__':
    # Create data
    train_data_path = os.getcwd() + train_data
    test_data_path = os.getcwd() + train_data

    X_train, y_train = CreateDataset(train_data_path)
    X_test, y_test = CreateDataset(test_data_path)
    X_data, y_data = CreateMergedXY([X_train, X_test], [y_train, y_test])

    KMeansClusteringAccuracy(X_data, y_data)
    IdentifyBestK(X_data.values)

    # By the elbow method, we pick K when at the point where the inertia stops dropping rapidly. As evidenced in the plot of the k-value vs. inertia, we see that the best k-value to pick is indeed 4.
    # The inertia drops from 1.6e11 to 1.3e11 from k=1 to k=4, and then from k=1.3e11 to 1.1e11 between k=4 and k=10, which means that the best k-value is 4.
    