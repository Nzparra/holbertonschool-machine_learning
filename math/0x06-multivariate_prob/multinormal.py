#!/usr/bin/env python3
""" Normal distribution """

import numpy as np


class MultiNormal():
    """ Normal distribution """

    def __init__(self, data):
        """ matrix data """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')
        d, n = data.shape
        self.mean = np.mean(data, axis=1, keepdims=True)
        X_mean = data - self.mean
        self.cov = np.dot(X_mean, X_mean.T) / (n - 1)

    def pdf(self, x):
        """ Returns the value of the PDF """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        d, _ = self.cov.shape
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        det = np.linalg.det(self.cov)
        first = 1 / (((2 * np.pi) ** (d / 2)) * (np.sqrt(det)))
        second = np.dot((x - self.mean).T, np.linalg.inv(self.cov))
        third = np.dot(second, (x - self.mean))
        pdf = first * np.exp((-1 / 2) * third)
        pdf = pdf[0][0]
        return pdf
