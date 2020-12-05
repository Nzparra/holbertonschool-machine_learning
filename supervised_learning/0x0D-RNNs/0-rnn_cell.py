#!/usr/bin/env python3
""" RNN cell """

import numpy as np


class RNNCell():
    """ RNN cell """
    def __init__(self, i, h, o):
        """
        * The weights will be used on the right side for matrix multiplication
        * The biases should be initialized as zeros
        """
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
            Returns: h_next, y
        """
        xh = np.concatenate((h_prev, x_t), axis=1)
        a_next = np.tanh(np.dot(xh, self.Wh) + self.bh)
        y_pred = np.dot(a_next, self.Wy) + self.by
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        return (a_next, y_pred)
