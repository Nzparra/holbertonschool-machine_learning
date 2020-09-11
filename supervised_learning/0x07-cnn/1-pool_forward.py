#!/usr/bin/env python3
"""  performs forward propagation over a
    convolutional layer of a neural network
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        Returns: the output of the convolutional layer
    """
    sh, sw = stride
    kh, kw = kernel_shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    conv_h = int(((h_prev - kh) / sh) + 1)
    conv_w = int(((w_prev - kw) / sw) + 1)
    conv = np.zeros((m, conv_h, conv_w, c_prev))
    for i in range(conv_h):
        for j in range(conv_w):
            st = i * sh
            end = (i * sh) + kh
            stw = j * sw
            endw = (j * sw) + kw
            X = A_prev[:, st:end, stw:endw]
            if mode == "max":
                Wx = np.max(X, axis=(1, 2))
            if mode == "avg":
                Wx = np.mean(X, axis=(1, 2))
            conv[:, i, j] = Wx
    return conv
