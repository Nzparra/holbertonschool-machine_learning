#!/usr/bin/env python3
"""  converts a numeric label vector into a one-hot matrix """

import numpy as np


def one_hot_encode(Y, classes):
    """
    Y is a numpy.ndarray with shape (m,) containing numeric class labels
        m is the number of examples
    classes is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m)
    """
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= Y.max():
        return None
    matrix = np.zeros((classes, len(Y)))
    for i in range(len(Y)):
        matrix[Y[i]][i] = 1
    return matrix
