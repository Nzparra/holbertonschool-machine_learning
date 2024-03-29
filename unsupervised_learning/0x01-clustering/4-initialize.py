#!/usr/bin/env python3
""" initialize  """

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
        Returns: pi, m, S, or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None)
    if type(k) is not int or k <= 0:
        return (None, None, None)
    n, d = X.shape
    m, _ = kmeans(X, k)
    pi = np.ones(k) / k
    ident = np.identity(d).reshape(-1)
    S = (np.tile(ident, k)).reshape(k, d, d)
    return (pi, m, S)
