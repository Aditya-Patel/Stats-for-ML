'''
Name: Aditya Patel
PUID: PATE1854
STAT 598 - Assignment 2
Summary: Implement the Gaussian Naive Bayes classifier and apply it to the spam dataset
'''

import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

path = os.getcwd() + '/spamData.txt'
smoothing_factor = 10 ** -9

def preprocessing(path, class_col):
        # convert text to pandas dataframe
        df = pd.read_csv(path, sep=' ', header=None)
        df[class_col] = df.iloc[:,57]
        df[class_col] = df[class_col].map({1:'spam', 0:'ham'})
        df.drop(columns=df.columns[57], inplace=True)
        return df

def reshapeXTestTrain(x_test, x_train):
    # Reshape the dataset in order to properly train and test the model
    return x_test.iloc[:,:-1], x_train.iloc[:,:-1]

class GaussianNaiveBayes:
    def calculatePriors(self, testset, class_col):
        # Calculate prior probability for both classes
        self.priors = testset.groupby(class_col).apply(lambda x: len(x)/testset.shape[0]).to_numpy()

    def calculateEpsilon(self, var_list, smoothing_factor):
        # Calculate smoothing parameter to ensure non-zero variances
        epsilon = (np.amax(var_list) * smoothing_factor)
        smoothed_var = var_list + epsilon
        return smoothed_var, epsilon

    def calculateMeanAndVar(self, testset, class_col, smoothing_factor=10**-9):
        # Calculate columnar means and variances for both classes
        self.mu = testset.groupby(class_col).mean().to_numpy()
        vars = testset.groupby(class_col).var(ddof=0).to_numpy()
        self.sig2, self.epsilon = self.calculateEpsilon(vars, smoothing_factor)

    def calculateGaussianProbability(self, x, classId):
        # Calculate the gaussian probability of a feature given its mean and variance
        # pdf = 1/SQRT(2 * pi * sig^2) * e^(-((x-mu)/(2*sig^2)))
        numer = np.exp(-((x-self.mu[classId])/(2 * self.sig2[classId])))
        denom = np.sqrt(2 * np.pi * self.sig2[classId])
        return (numer/denom)

    def calculatePosteriorProbability(self, x, class_dict):
        posteriorProbs = []
        for i in [0, 1]:
            priorProb = np.log(self.priors[i])
            conditionalProb = np.sum(np.log(self.calculateGaussianProbability(x, i)))
            posteriorProbs.append(priorProb + conditionalProb)
        return class_dict[np.argmax(posteriorProbs)]

    def fit(self, X_train, fieldname, smoothing_factor=10**-9):
        self.calculateMeanAndVar(X_train, fieldname, smoothing_factor)
        self.calculatePriors(X_train, fieldname)
        
    def predict(self, X_test):
        preds = []
        for f in X_test.to_numpy():
            fPred = self.calculatePosteriorProbability(f, {1:'spam', 0:'ham'})
            preds.append(fPred)
        return preds


if __name__ == '__main__':
    GNB = GaussianNB()
    
    X = preprocessing(path, 'email_type')
    y = X.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.75)

    X_test = X_test.iloc[:,:-1]

    GNB.fit(X_test, y_test)
    homebrewGNB = GaussianNaiveBayes()
    homebrewGNB.fit(X_train, 'email_type')

    sk_pred = GNB.predict(X_test)
    y_pred = homebrewGNB.predict(X_test)

    hbPrec, hbRecall, hbF1, hbSupport = precision_recall_fscore_support(y_test, y_pred)
    hbAcc = accuracy_score(y_test, y_pred)
    skPrec, skRecall, skF1, skSupport = precision_recall_fscore_support(y_test, sk_pred)
    skAcc = accuracy_score(y_test, sk_pred)

    print("Testing Metrics:")
    print("Self-coded GNB:\n\tAccuracy:\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}".format(hbAcc, hbPrec, hbRecall, hbF1))
    print("SK Learn GNB:\n\tAccuracy:\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}".format(skAcc, skPrec, skRecall, skF1))





