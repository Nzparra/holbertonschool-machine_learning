#!/usr/bin/env python3
"""  converts a numeric label vector into a one-hot matrix """

import numpy as np


def one_hot_decode(one_hot):
    """
        one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
        classes is the maximum number of classes
        m is the number of examples
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    decode = np.zeros(one_hot.shape[1], dtype=int)
    for i, j in enumerate(one_hot.transpose()):
        if sum(j) > 1:
            return None
        for ij, ik in enumerate(j):
            if (ik) > 1 or (ik != 0 and ik != 1):
                return None
            if ij == 1:
                decode[i] = ij
    return decode
