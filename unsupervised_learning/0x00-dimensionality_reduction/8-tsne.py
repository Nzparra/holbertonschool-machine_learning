#!/usr/bin/env python3
""" Stochastic Neighbor Embedding """

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0,
         iterations=1000, lr=500):
    """
    Returns: Y, a numpy.ndarray of shape (n, ndim) containing
    the optimized low dimensional transformation of X
    """
    n, d = X.shape
    PCA = pca(X, idims)
    P = P_affinities(X=PCA, perplexity=perplexity)
    Y = np.random.randn(n, ndims)
    iY = Y
    P = 4 * P
    for i in range(0, iterations):
        dY, Q = grads(Y, P)
        if i < 20:
            alpha = 0.5
        else:
            alpha = 0.8
        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            a = 'Cost at iteration {}: {}'.format(i + 1, C)
            print(a)
        temp = Y
        Y = Y - (lr * dY) + (alpha * (Y - iY))
        iY = temp
        if (i + 1) == 100:
            P = P / 4
    return Y
