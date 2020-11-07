#!/usr/bin/env python3
""" Centroids """

import numpy as np


def kmeans(X, k, iterations=1000):
    """
        Returns: C, clss, or None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None)
    if type(k) is not int or k <= 0:
        return (None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None)
    n, d = X.shape
    min_X = np.min(X, axis=0).astype(np.float)
    max_X = np.max(X, axis=0).astype(np.float)
    centr = np.random.uniform(low=min_X, high=max_X, size=(k, d))
    for i in range(iterations):
        copy = centr.copy()
        data = X[:, np.newaxis, :]
        aux_centr = copy[np.newaxis, :, :]
        dist = (np.square(data - aux_centr)).sum(axis=2)
        clase = np.argmin(dist, axis=1)
        for i in range(k):
            mask = np.where(clase == i)
            new_data = X[mask]
            if len(new_data) == 0:
                copy[i] = np.random.uniform(min_X, max_X, (1, d))
            else:
                copy[i] = np.mean(new_data, axis=0)
        if (centr == copy).all():
            break
        else:
            centr = copy
    data = X[:, np.newaxis, :]
    aux_centr = centr[np.newaxis, :, :]
    dist = (np.square(data - aux_centr)).sum(axis=2)
    clase = np.argmin(dist, axis=1)
    return (centr, clase)
