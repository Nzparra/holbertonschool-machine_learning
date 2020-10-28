#!/usr/bin/env python3
""" all variables required to calculate the P affinities in t-SNE """

import numpy as np


def P_init(X, perplexity):
    """ Returns: (D, P, betas, H) """
    n, d = X.shape
    X_square = np.sum(np.square(X), axis=1)
    Y_square = np.sum(np.square(X), axis=1)
    XY = np.dot(X, X.T)
    D = np.add(np.add((-2 * XY), X_square).T, Y_square)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return (D, P, betas, H)
