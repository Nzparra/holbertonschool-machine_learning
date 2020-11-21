#!/usr/bin/env python3
""" Bayes Optimization """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    Bayes Optimization using Gaussian Process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01,
                 minimize=True):
        """
          minimize: a bool for minimization versus maximization
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
            Uses the Expected Improvement acquisition function
            Returns: X_next, EI
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
        index = np.argmax(ei)
        best_sample = self.X_s[index]
        return (best_sample, ei)
