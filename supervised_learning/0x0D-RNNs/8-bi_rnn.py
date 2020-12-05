#!/usr/bin/env python3
"""
bidirectional RNN forward propagation
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Returns: H, Y
    """
    t, m, i = X.shape
    m, h = h_0.shape
    HF = np.zeros((t, m, h))
    HB = np.zeros((t, m, h))
    h_prev_p = h_0
    h_next_b = h_t
    for i in range(t):
        x_tf = X[i]
        x_tb = X[-(i + 1)]
        h_prev_p = bi_cell.forward(h_prev_p, x_tf)
        h_next_b = bi_cell.backward(h_next_b, x_tb)
        HF[i] = h_prev_p
        HB[-(i + 1)] = h_next_b
    H = np.concatenate((HF, HB), axis=-1)
    Y = bi_cell.output(H)
    return (H, Y)
