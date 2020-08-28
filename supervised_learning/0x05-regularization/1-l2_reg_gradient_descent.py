#!/usr/bin/env python3
"""
    calculates the cost of a neural network with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        Y is a one-hot numpy.ndarray of shape (classes, m)
        classes is the number of classes
        m is the number of data points
        weights is a dictionary of the weights and biases
        cache is a dictionary of the outputs of each layer
        alpha is the learning rate
        lambtha is the L2 regularization parameter
        L is the number of layers of the network
        The neural network uses tanh activations on each layer
        The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    temp = weights.copy()
    d = 0
    for i in range(L, 0, -1):
        if i == L:
            d = cache["A{}".format(i)] - Y
            dw = np.matmul(d, cache["A{}".format(i - 1)].transpose()) / m
            dwl = dw + ((lambtha / m) * temp["W{}".format(i)])
            db = (d.sum(axis=1, keepdims=True)) / m
        else:
            w = temp["W{}".format(i + 1)]
            dzl = np.matmul(w.transpose(), d) * (1 -
                                                 (cache["A{}".format(i)]) ** 2)
            dw = (np.matmul(dzl, cache["A{}".format(i - 1)].transpose())) / m
            dwl = dw + ((lambtha / m) * temp["W{}".format(i)])
            db = (dzl.sum(axis=1, keepdims=True)) / m
            d = dzl
        weights["W{}".format(i)] = temp["W{}".format(i)] - (alpha * (dwl))
        weights["b{}".format(i)] = temp["b{}".format(i)] - (alpha * db)
