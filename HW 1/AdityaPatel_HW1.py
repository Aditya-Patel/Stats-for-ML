"""
Aditya Patel
Homework 1
Stat 598 - Statistical Machine Learning 
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as lm


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
    
    # Locations
    tj = [0.1,0.13,0.15,0.23,0.25,0.40,0.44,0.65,0.76,0.78,0.81]
    
    # Heights
    h = [4,5,3,4,5,4.2,2.1,4.3,3.1,5.1,4.2]
    
    # Weights
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

# Create initial plots
f_bumps = bumps(t, N)
f_doppler = doppler(t, N)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(t, f_bumps)
plt.title('Bumps function')
plt.xlabel('t')
plt.ylabel('f(t)')

plt.subplot(1, 2, 2)
plt.plot(t, f_doppler)
plt.title('Doppler function')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.show()

# Add gaussian white noise and rescale to SNR of 7
sigma = 1
bump_scale = 10
dopp_scale = 25

gaussian_white_noise = np.random.normal(0, sigma, N)
noisy_bumps = (bump_scale * f_bumps) + gaussian_white_noise
noisy_doppler = (dopp_scale * f_doppler) + gaussian_white_noise

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(t, noisy_bumps)
plt.title('Noisy Bumps function')
plt.xlabel('t')
plt.ylabel('f(t)')

plt.subplot(1, 2, 2)
plt.plot(t, noisy_doppler)
plt.title('Noisy Doppler function')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.show()

# Polynomial regression
M = 4
poly = PolynomialFeatures(degree=M, include_bias=False)

poly_bumps = poly.fit_transform(t.reshape(-1,1))
poly_reg_bumps = lm.LinearRegression()
poly_reg_bumps.fit(poly_bumps, noisy_bumps)
poly_bumps_pred = poly_reg_bumps.predict(poly_bumps)

poly_dopp = poly.fit_transform(t.reshape(-1,1))
poly_reg_dopp = lm.LinearRegression()
poly_reg_dopp.fit(poly_dopp, noisy_doppler)
poly_dopp_pred = poly_reg_dopp.predict(poly_dopp)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(t, bump_scale*f_bumps, c='black', label='Scaled function')
plt.scatter(t,noisy_bumps, c='gray', label = 'Noisy function')
plt.plot(t, poly_bumps_pred, c='red', label = 'Polynomial Regression (M={})'.format(M))
plt.legend()
plt.xlabel('t')
plt.ylabel('f(t)')

plt.subplot(1, 2, 2)
plt.plot(t, dopp_scale * f_doppler, c='black', label='Scaled function')
plt.scatter(t, noisy_doppler, c='gray', label = 'Noisy function')
plt.plot(t, poly_dopp_pred, c='red', label = 'Polynomial Regression (M={})'.format(M))
plt.legend()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.show()