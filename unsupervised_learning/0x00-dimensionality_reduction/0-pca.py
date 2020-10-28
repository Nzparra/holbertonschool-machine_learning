#!/usr/bin/env python3
""" Getting weights of PCA """

import numpy as np


def pca(X, var=0.95):
    """
    Returns: the weights matrix, W, that maintains var fraction
    of X‘s original variance
    """
    U, S, VT = np.linalg.svd(X)
    cumsum_array = np.cumsum(S)
    threshold = cumsum_array[-1] * var
    mask = np.where(cumsum_array < threshold)
    r = len(cumsum_array[mask])
    W = VT.T
    Wr = W[:, :r+1]
    return Wr
