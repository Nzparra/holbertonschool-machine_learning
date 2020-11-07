#!/usr/bin/env python3
""" Initialize"""

import numpy as np


def initialize(X, k):
    """
        Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    _, d = X.shape
    min_Xvals = np.min(X, axis=0).astype(np.float)
    max_Xvals = np.max(X, axis=0).astype(np.float)
    centroids = np.random.uniform(low=min_Xvals, high=max_Xvals, size=(k, d))
    return centroids
