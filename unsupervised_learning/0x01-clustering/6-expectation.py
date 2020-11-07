#!/usr/bin/env python3
""" Expectation """

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
        Returns: posterior, prob_gmm, or None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None)
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return (None, None)
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return (None, None)
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return (None, None)
    mask1 = np.where(pi < 0, True, False)
    mask2 = np.where(pi > 1, True, False)
    if mask1.any() or mask2.any():
        return (None, None)
    if X.shape[1] != S.shape[1] or S.shape[1] != S.shape[2]:
        return (None, None)
    if X.shape[1] != m.shape[1]:
        return (None, None)
    if m.shape[0] != S.shape[0]:
        return (None, None)
    if pi.shape[0] != m.shape[0]:
        return (None, None)
    n, _ = X.shape
    k, d, _ = S.shape
    g = np.zeros((k, n))
    for clus in range(k):
        prob = pdf(X, m[clus], S[clus])
        prior = pi[clus]
        g[clus] = prior * prob
    total = np.sum(g, axis=0, keepdims=True)
    posterior = g / total
    prob_gmm = np.sum(np.log(total))
    return (posterior, prob_gmm)
