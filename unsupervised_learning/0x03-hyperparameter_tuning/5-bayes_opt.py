#!/usr/bin/env python3
""" Gaussian process """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ Performs Bayesian optimization """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        X_s = np.linspace(min, max, ac_samples)
        self.X_s = (np.sort(X_s)).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
            Calculates the next best sample location
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is True:
            optimize = np.amin(self.gp.Y)
            imp = optimize - mu - self.xsi
        else:
            optimize = np.amax(self.gp.Y)
            imp = mu - optimize - self.xsi
        Z = np.zeros(sigma.shape[0])
        for i in range(sigma.shape[0]):
            if sigma[i] != 0:
                Z[i] = imp[i] / sigma[i]
            else:
                Z[i] = 0
        ei = np.zeros(sigma.shape)
        for i in range(len(sigma)):
            if sigma[i] > 0:
                ei[i] = imp[i] * norm.cdf(Z[i]) + sigma[i] * norm.pdf(Z[i])
            else:
                ei[i] = 0
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei

    def optimize(self, iterations=100):
        """
        Returns: X_opt, Y_opt
        """
        for i in range(iterations):
            x_new, _ = self.acquisition()
            if [x_new] in self.gp.X:
                break
            y_new = self.f(x_new)
            self.gp.update(x_new, y_new)
        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        x_new = self.gp.X[index]
        y_new = self.gp.Y[index]
        return x_new, y_new
