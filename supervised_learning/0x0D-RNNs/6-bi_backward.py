#!/usr/bin/env python3
""" Bidirectional Cell"""

import numpy as np


class BidirectionalCell():
    """ Bidirectional Cell"""

    def __init__(self, i, h, o):
        """
        * The weights will be used on the right side for matrix
          multiplication
        * The biases should be initialized as zeros
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        * Returns: h_next, the next hidden state
        """
        xh = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(xh, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        * Returns: h_pev, the previous hidden state
        """
        xh = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(xh, self.Whb) + self.bhb)
        return h_prev
