#!/usr/bin/env python3
""" simple RNN """

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
        Returns: H, Y
    """
    t, _, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    H[0] = h_0
    h_prev = h_0
    for i in range(t):
        x_t = X[i]
        h_prev, y_pred = rnn_cell.forward(h_prev, x_t)
        H[i + 1] = h_prev
        Y[i] = y_pred
    return (H, Y)
