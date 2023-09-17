"""
Aditya Patel
Homework 1
Stat 598 - Statistical Machine Learning 
"""
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
import sklearn.linear_model as lm

# Define Number of Replications:
repl = 20

# Define t_i
N = 2048
t = np.zeros(N)

for i in range(N):
    try:
        t[i] = i/N
    except(IndexError):
        print(i)


# Define Doppler and Bump functions
def bumps(t, N):
    # Output array
    f = np.zeros(shape=N)
    
    # Locations, heights, and weights from Donoho et al.
    tj = [0.1,0.13,0.15,0.23,0.25,0.40,0.44,0.65,0.76,0.78,0.81]
    h = [4,5,3,4,5,4.2,2.1,4.3,3.1,5.1,4.2]
    w = [0.005,0.005,0.006,0.01,0.01,0.03,0.01,0.01,0.005,0.008,0.005]
    
    # K function
    def K(t):
        return (1+abs(t)) ** -4
    
    # Iterate through N values
    for i in range(N):
        # Iterate from j = 1 to 11
        for j in range(0, 11):
            t_prime = (t[i] - tj[j])/w[j]
            f[i]+= h[j]*K(t_prime)
    
    return f

def doppler(t, N):
    f = np.zeros(shape=N)

    def dopplerFxn(t):
        root = math.sqrt(t*(1-t))
        sinusoid = math.sin((2.1*math.pi)/(t + 0.05))
        return root * sinusoid
    
    for i in range(N):
        f[i] = dopplerFxn(t[i])

    return f

# Define RMSE function
def loss_fxn(f, f_hat, N):
    return (1 / np.sqrt(N)) * np.linalg.norm(f_hat - f, ord=2)

# Generate output for Doppler and Bumps functions
f_bumps = bumps(t, N)
f_doppler = doppler(t, N)

poly_loss_bumps = []
knn_loss_bumps = []
poly_bumps_rmse = []
knn_bumps_rmse =[]

poly_loss_dopp = []
knn_loss_dopp = []
poly_dopp_rmse = []
knn_dopp_rmse = []

# Create replications and iterate
for r in range(repl):
    # Add gaussian white noise and rescale to SNR of 7
    sigma = 1 # Identified from Donoho et al.
    bump_scale = 10 # Estimated
    dopp_scale = 25 # Estimated

    gaussian_noise = np.random.normal(0, sigma, N)
    noisy_bumps = (bump_scale * f_bumps) + gaussian_noise
    noisy_doppler = (dopp_scale * f_doppler) + gaussian_noise

    # Polynomial regression
    M = 4
    poly = PolynomialFeatures(degree=M, include_bias=False)

    poly_bump_train = poly.fit_transform(t.reshape(-1,1))
    poly_dopp_train = poly.fit_transform(t.reshape(-1,1))

    poly_reg_bumps = lm.LinearRegression()
    poly_reg_dopp = lm.LinearRegression()

    poly_reg_bumps.fit(poly_bump_train, noisy_bumps)
    poly_reg_dopp.fit(poly_dopp_train, noisy_doppler)

    poly_bumps_pred = poly_reg_bumps.predict(poly_bump_train)
    poly_dopp_pred = poly_reg_dopp.predict(poly_dopp_train)

    # K-Nearest Neighbors Regression
    k = 5
    bumps_neigh = KNeighborsRegressor(n_neighbors=k)
    dopp_neigh = KNeighborsRegressor(n_neighbors=k)

    bumps_neigh.fit(t.reshape(-1,1), noisy_bumps)
    dopp_neigh.fit(t.reshape(-1,1), noisy_doppler)

    knn_bumps_pred = bumps_neigh.predict(t.reshape(-1,1))
    knn_dopp_pred = dopp_neigh.predict(t.reshape(-1,1))

    # RMSE-Loss Evaluation
    poly_loss_bumps.append(loss_fxn(bump_scale * f_bumps, poly_bumps_pred, N))
    knn_loss_bumps.append(loss_fxn(bump_scale * f_bumps, knn_bumps_pred, N))
    poly_bumps_rmse.append(mean_squared_error((bump_scale * f_bumps), poly_bumps_pred, squared=False))
    knn_bumps_rmse.append(mean_squared_error((bump_scale * f_bumps), knn_bumps_pred, squared=False))
    
    poly_loss_dopp.append(loss_fxn(dopp_scale * f_doppler, poly_dopp_pred, N))
    knn_loss_dopp.append(loss_fxn(dopp_scale * f_doppler, knn_dopp_pred, N))
    poly_dopp_rmse.append(mean_squared_error((dopp_scale * f_doppler), poly_dopp_pred, squared=False))
    knn_dopp_rmse.append(mean_squared_error((dopp_scale * f_doppler), knn_dopp_pred, squared=False))

arl_poly_bumps = np.mean(poly_loss_bumps)
arl_knn_bumps = np.mean(knn_loss_bumps)
armse_poly_bumps = np.mean(poly_bumps_rmse)
armse_knn_bumps = np.mean(knn_bumps_rmse)

arl_poly_dopp = np.mean(poly_loss_dopp)
arl_knn_dopp = np.mean(knn_loss_dopp)
armse_poly_dopp = np.mean(poly_dopp_rmse)
armse_knn_dopp = np.mean(knn_dopp_rmse)

print("Average Root Loss Over 20 Replications:")
print("Bumps Function:")
print("\tPolynomial Regression:\t{:.4f}".format(arl_poly_bumps))
print("\tK-Nearest-Neighbors Regression:\t{:.4f}".format(arl_knn_bumps))
print("Doppler Function:")
print("\tPolynomial Regression:\t{:.4f}".format(arl_poly_dopp))
print("\tK-Nearest-Neighbors Regression:\t{:.4f}".format(arl_knn_dopp))

print("\nAverage Root Mean Squared Error Over 20 Replications:")
print("Bumps Function:")
print("\tPolynomial Regression:\t{:.4f}".format(armse_poly_bumps))
print("\tK-Nearest-Neighbors Regression:\t{:.4f}".format(armse_knn_bumps))
print("Doppler Function:")
print("\tPolynomial Regression:\t{:.4f}".format(armse_poly_dopp))
print("\tK-Nearest-Neighbors Regression:\t{:.4f}".format(armse_knn_dopp))

print("\nAverage RMSE Over Average Root Loss for 20 Replications:")
print("Bumps Function:")
print("\tPolynomial Regression:\t{:.4f}".format(armse_poly_bumps/arl_poly_bumps))
print("\tK-Nearest-Neighbors Regression:\t{:.4f}".format(armse_knn_bumps/arl_knn_bumps))
print("Doppler Function:")
print("\tPolynomial Regression:\t{:.4f}".format(armse_poly_dopp/arl_poly_dopp))
print("\tK-Nearest-Neighbors Regression:\t{:.4f}".format(armse_knn_dopp/arl_knn_dopp))

# Plotting
plt.figure()
plt.subplot(2, 1, 1)
plt.title('Regression fits for Bumps function')
plt.scatter(t, noisy_bumps, c='gray', s=0.25, label = 'Training Data (SNR = 7)')
plt.plot(t, poly_bumps_pred, c='red', label = 'Polynomial Regression (M={})'.format(M))
plt.plot(t, knn_bumps_pred, c='lime', label = 'K-Nearest Neighbors Regression (N={})'.format(k))
plt.plot(t, bump_scale*f_bumps, c='black', label='Scaled function')
plt.legend()
plt.xlabel('t')
plt.ylabel('f(t)')

plt.subplot(2, 1, 2)
plt.title('Regression fits for Doppler function')
plt.scatter(t, noisy_doppler, s=0.25, c='gray', label = 'Training Data (SNR = 7)')
plt.plot(t, poly_dopp_pred, c='red', label = 'Polynomial Regression (M={})'.format(M))
plt.plot(t, knn_dopp_pred, c='lime', label = 'K-Nearest Neighbors Regression (N={})'.format(k))
plt.plot(t, dopp_scale*f_doppler, c='black', label='Scaled function')
plt.legend()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.show()