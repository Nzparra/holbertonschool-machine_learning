#!/usr/bin/env python3
""" Getting weights of PCA """

import numpy as np


def pca(X, ndim):
    """
    Returns: T, a numpy.ndarray of shape (n, ndim) containing
    the transformed version of X
    """
    n, d = X.shape
    X_mean = X - np.mean(X, axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(X_mean, full_matrices=False)
    W = VT.T
    Wr = W[:, :ndim]
    Tr = np.dot(X_mean, Wr)
    return Tr
