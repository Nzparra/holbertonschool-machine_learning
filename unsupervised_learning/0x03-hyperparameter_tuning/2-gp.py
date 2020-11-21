#!/usr/bin/env python3
""" Prediction """

import numpy as np


class GaussianProcess():
    """ Gaussian Process """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
            Returns: the covariance kernel matrix as a numpy.ndarray of
            shape (m, n)
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        ''' Returns Covariance matrix (m x n) '''
        first = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        second = np.sum(X2 ** 2, axis=1)
        third = -2 * np.dot(X1, X2.T)
        sqdist = first + second + third
        kernel_1 = (self.sigma_f ** 2)
        kernel_2 = np.exp(-0.5 / self.l ** 2 * sqdist)
        kernel = kernel_1 * kernel_2
        return kernel

    def predict(self, X_s):
        """ Returns: mu, sigma """
        K = (self.K)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = mu_s.reshape(-1)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        var_s = np.diag(cov_s)
        return (mu_s, var_s)

    def update(self, X_new, Y_new):
        """
            Updates the public instance attributes X, Y, and K
        """
        self.X = np.append(self.X, X_new.reshape(-1, 1), axis=0)
        self.Y = np.append(self.Y, Y_new.reshape(-1, 1), axis=0)
        self.K = self.kernel(self.X, self.X)
