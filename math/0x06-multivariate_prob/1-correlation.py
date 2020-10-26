#!/usr/bin/env python3
"""
correlation matrix
"""

import numpy as np


def correlation(C):
    """ Returns the correlation matrix """
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')
    d, _ = C.shape
    variance = np.diag(C).reshape(1, -1)
    stddev = np.sqrt(variance)
    matrix_std = np.dot(stddev.T, stddev)
    corr = C / matrix_std
    return corr
