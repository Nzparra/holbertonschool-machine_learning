#!/usr/bin/env python3
""" Stochastic Neighbor Embedding """

import numpy as np


def Q_affinities(Y):
    """ Returns: Q, num """
    n, d = Y.shape
    x_square = np.sum(np.square(Y), axis=1)
    y_square = np.sum(np.square(Y), axis=1)
    xy = np.dot(Y, Y.T)
    D = np.add(np.add((-2 * xy), x_square).T, y_square)
    Q = np.zeros((n, n))
    num = np.zeros((n, n))
    for i in range(n):
        Di = D[i].copy()
        Di = np.delete(Di, i, axis=0)
        numerator = (1 + Di) ** (-1)
        numerator = np.insert(numerator, i, 0)
        num[i] = numerator
    den = num.sum()
    Q = num / den
    return Q, num
