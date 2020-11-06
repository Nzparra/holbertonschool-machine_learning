#!/usr/bin/env python3
""" Likelihood """

import numpy as np


def likelihood(x, n, P):
    """
    Returns: a 1D numpy.ndarray containing the likelihood
    of obtaining the data, x and n, for each probability
    in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        m = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(m)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    for elem in P:
        if not (elem >= 0 and elem <= 1):
            a = 'All values in P must be in the range [0, 1]'
            raise ValueError(a)
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)
    combination = fact_n / (fact_x * fact_nx)
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
