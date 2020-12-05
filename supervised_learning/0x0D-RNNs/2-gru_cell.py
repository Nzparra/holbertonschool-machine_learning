#!/usr/bin/env python3
""" GRU      """

import numpy as np


class GRUCell():
    """ Structure GRU cell """
    def __init__(self, i, h, o):
        """
        * The weights will be used on the right side for matrix
          multiplication
        * The biases should be initialized as zero
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
            Returns: h_next, y
        """
        xh = np.concatenate((h_prev, x_t), axis=1)
        rt = np.dot(xh, self.Wr) + self.br
        rt = 1 / (1 + np.exp(-rt))
        zt = np.dot(xh, self.Wz) + self.bz
        zt = 1 / (1 + np.exp(-zt))
        r_h = rt * h_prev
        r_hx = np.concatenate((r_h, x_t), axis=1)
        h_hat = np.tanh(np.dot(r_hx, self.Wh) + self.bh)
        h_next = zt * h_hat + (1 - zt) * h_prev
        y_pred = np.dot(h_next, self.Wy) + self.by
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        return (h_next, y_pred)
