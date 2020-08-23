#!/usr/bin/env python3
"""  Adam optimization algorithm """


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
        loss is the loss of the network
        alpha is the learning rate
        beta1 is the weight used for the first moment
        beta2 is the weight used for the second moment
        epsilon is a small number to avoid division by zero
        Returns: the Adam optimization operation
    """
    V = (beta1 * v) + ((1 - beta1) * grad)
    S = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    V1 = V / (1 - (beta1 ** t))
    S1 = S / (1 - (beta2 ** t))
    var = var - (alpha * (V1 / ((S1 ** (1/2)) + epsilon)))
    return (var, V, S)
