#!/usr/bin/env python3
""" Stochastic Neighbor Embedding """

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """ Returns: (dY, Q) """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    equation1 = ((P - Q) * num)
    dY = np.zeros((n, ndim))
    for i in range(n):
        aux = np.tile(equation1[:, i].reshape(-1, 1), ndim)
        dY[i] = (aux * (Y[i] - Y)).sum(axis=0)
    return (dY, Q)
