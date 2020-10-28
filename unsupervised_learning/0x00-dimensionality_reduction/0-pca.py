#!/usr/bin/env python3
""" Getting weights of PCA """

import numpy as np


def pca(X, var=0.95):
    """
    Returns: the weights matrix, W, that maintains var fraction
    of X‘s original variance
    """
    U, S, VT = np.linalg.svd(X)
    t_variance = np.cumsum(S) / np.sum(S)
    r = np.argwhere(t_variance >= var)[0, 0]
    W = VT[:r + 1].T
    return W
