#!/usr/bin/env python3
""" variance """

import numpy as np


def variance(X, C):
    """
        Returns: var, or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    if C.shape[0] > X.shape[0]:
        return None
    n, _ = X.shape
    data = X[:, np.newaxis, :]
    centr = C[np.newaxis, :, :]
    dist = (np.square(data - centr)).sum(axis=2)
    mini_per_datapoint = np.amin(dist, axis=1)
    var = np.sum(mini_per_datapoint)
    return var
