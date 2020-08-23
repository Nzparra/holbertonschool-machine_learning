#!/usr/bin/env python3
"""  normalizes an unactivated output of a neural network """


def batch_norm(Z, gamma, beta, epsilon):
    """
        Z is a numpy.ndarray of shape (m, n) that should be normalized
        m is the number of data points
        n is the number of features in Z
        gamma is a numpy.ndarray of shape (1, n)
        beta is a numpy.ndarray of shape (1, n)
        epsilon is a small number used to avoid division by zero
        Returns: the normalized Z matrix
    """
    med = Z.sum(axis=0) / Z.shape[0]
    var = (Z - med) ** 2
    var = var.sum(axis=0) / Z.shape[0]
    Z_ = (Z - med) / ((var + epsilon) ** (1/2))
    Z_ = (gamma * Z_) + beta
    return Z_
