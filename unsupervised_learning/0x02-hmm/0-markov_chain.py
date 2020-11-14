#!/usr/bin/env python3
""" markov chain """

import numpy as np


def markov_chain(P, s, t=1):
    """
        Returns: a numpy.ndarray of shape (1, n)
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    if P.shape[0] != P.shape[1] or s.shape[0] != 1:
        return None
    if P.shape[0] != s.shape[1]:
        return None
    if type(t) is not int or t <= 0:
        return None
    sum_test = np.sum(P, axis=1)
    for elem in sum_test:
        if not np.isclose(elem, 1):
            return None
    sum_test = np.sum(s)
    if not np.isclose(sum_test, 1):
        return None
    prob = s.copy()
    for i in range(t):
        prob = np.matmul(prob, P)
    return prob
