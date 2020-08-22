#!/usr/bin/env python3
""" shuffles the data points in two matrices the same way """

import numpy as np


def shuffle_data(X, Y):
    """
        m is the number of data points
        nx is the number of features
    """
    perm = np.random.permutation(np.arange(X.shape[0]))
    return X[perm], Y[perm]
