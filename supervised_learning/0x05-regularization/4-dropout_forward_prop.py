#!/usr/bin/env python3
"""
    calculates the cost of a neural network with L2 regularization
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation function that should be used on the layer
        lambtha is the L2 regularization parameter
        Returns: the output of the new layer
    """
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        Z = np.matmul(weights["W{}".format(i)], cache["A{}".format(i - 1)])
        Z = Z + weights["b{}".format(i)]
        if i != L:
            D = np.random.binomial(1, keep_prob, size=Z.shape)
            cache["D{}".format(i)] = D
            A = (D / keep_prob) * np.tanh(Z)
        else:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        cache["A{}".format(i)] = A
    return cache
