#!/usr/bin/env python3
""" estimation and maximization """

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
        Returns: pi, m, S, g, l, or None, None, None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None, None, None)
    if type(k) is not int or k <= 0:
        return (None, None, None, None, None)
    if type(iterations)is not int or iterations <= 0:
        return (None, None, None, None, None)
    if type(tol) is not float or tol < 0:
        return (None, None, None, None, None)
    if type(verbose) is not bool:
        return (None, None, None, None, None)
    pi, m, S = initialize(X, k)
    l_init = 0
    count = 0
    g, log_like = expectation(X, pi, m, S)
    while(count < iterations):
        if (np.abs(l_init - log_like)) <= tol:
            break
        if verbose is True and count % 10 == 0:
            m1 = 'Log Likelihood after {}'.format(count)
            m2 = ' iterations: {}'.format(log_like.round(5))
            print(m1 + m2)
        l_init = log_like
        pi, m, S = maximization(X, g)
        g, log_like = expectation(X, pi, m, S)
        count += 1
    if verbose is True:
        m1 = 'Log Likelihood after {}'.format(count)
        m2 = ' iterations: {}'.format(log_like.round(5))
        print(m1 + m2)
    return (pi, m, S, g, log_like)
