#!/usr/bin/env python3
"""
    calculates the cost of a neural network with L2 regularization
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation function that should be used on the layer
        lambtha is the L2 regularization parameter
        Returns: the output of the new layer
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count = count + 1
    if count == patience:
        return (True, count)
    else:
        return(False, count)
