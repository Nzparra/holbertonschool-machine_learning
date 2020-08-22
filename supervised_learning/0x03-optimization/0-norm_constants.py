#!/usr/bin/env python3
""" Normalization Constants """

import numpy as np


def normalization_constants(X):
    """
        m is the number of data points
        nx is the number of features
    """
    mean = X.sum(axis=0) / X.shape[0]
    var = ((X - mean) ** 2).sum(axis=0) / X.shape[0]
    return (mean, np.sqrt(var))
