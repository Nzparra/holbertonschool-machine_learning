#!/usr/bin/env python3
""" Gradient Descent """

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
        The weights of the network should be updated in place
    """
    aux = weights.copy()
    dZ = 0
    m = Y.shape[1]
    for i in range(L, 0, -1):
        key_w = "W{}".format(i)
        key_b = "b{}".format(i)
        key_in = "A{}".format(i)
        X = "A{}".format(i - 1)
        if i == L:
            dZ = (cache[key_in] - Y)
            dW = np.matmul(dZ, cache[X].transpose()) / m
            db = (dZ.sum(axis=1, keepdims=True)) / m
        else:
            deriv = 1 - ((cache[key_in]) * (cache[key_in]))
            weight = aux["W{}".format(i + 1)]
            dropout = (cache["D{}".format(i)] / keep_prob) * deriv
            dZL = np.matmul(weight.transpose(), dZ) * dropout
            dW = (np.matmul(dZL, cache[X].transpose())) / m
            db = (dZL.sum(axis=1, keepdims=True)) / m
            dZ = dZL
        weights[key_w] = aux[key_w] - (alpha * (dW))
        weights[key_b] = aux[key_b] - (alpha * db)
