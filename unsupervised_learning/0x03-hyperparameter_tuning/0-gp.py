#!/usr/bin/env python3
""" Init """

import numpy as np


class GaussianProcess():
    """
    Gaussian Process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
            Returns: the covariance kernel matrix as a numpy.ndarray
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        ''' Returns Covariance matrix (m x n)'''
        first = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        second = np.sum(X2 ** 2, axis=1)
        third = -2 * np.dot(X1, X2.T)
        sqdist = first + second + third
        kernel_1 = (self.sigma_f ** 2)
        kernel_2 = np.exp(-0.5 / self.l ** 2 * sqdist)
        kernel = kernel_1 * kernel_2
        return kernel
