#!/usr/bin/env python3
"""  performs forward propagation over a
    convolutional layer of a neural network
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
        Returns: the output of the convolutional layer
    """
    sh, sw = stride
    kh, kw, c_prev, c_aft = W.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    if padding != "same":
        ph, pw = (0, 0)
    else:
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    new = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    conv_h = int(((h_prev + (2 * ph) - kh) / sh) + 1)
    conv_w = int(((w_prev + (2 * pw) - kw) / sw) + 1)
    conv = np.zeros((m, conv_h, conv_w, c_aft))
    for i in range(conv_h):
        for j in range(conv_w):
            for k in range(c_prev):
                st = i * sh
                end = (i * sh) + kh
                stw = j * sw
                endw = (j * sw) + kw
                Wx = (W[:, :, :, k] * (new[:, st:end, stw:endw]))
                Wx = Wx.sum(axis=(1, 2, 3))
                conv[:, i, j, k] = Wx
    return activation(conv + b)
