#!/usr/bin/env python3
""" maximization  """

import numpy as np


def maximization(X, g):
    """
        Returns: pi, m, S, or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None)
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return (None, None, None)
    if X.shape[0] != g.shape[1]:
        return (None, None, None)
    sum = np.sum(g, axis=0)
    sum = np.sum(sum)
    if (int(sum) != X.shape[0]):
        return (None, None, None)
    n, d = X.shape
    k, _ = g.shape
    N_soft = np.sum(g, axis=1)
    pi = N_soft / n
    mean = np.zeros((k, d))
    cov = np.zeros((k, d, d))
    for clus in range(k):
        rik = g[clus]
        denomin = N_soft[clus]
        mean[clus] = np.matmul(rik, X) / denomin
        first = rik * (X - mean[clus]).T
        cov[clus] = np.matmul(first, (X - mean[clus])) / denomin
    return (pi, mean, cov)
