#!/usr/bin/env python3
"""  performs forward propagation over a
    convolutional layer of a neural network
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        Returns: the output of the convolutional layer
    """
    sh, sw = stride
    kh, kw = kernel_shape
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    conv = np.zeros(A_prev.shape)
    for i in range(m):
        ima = A_prev[i]
        for j in range(h_new):
            for k in range(w_new):
                for l in range(c_new):
                    st = j * sh
                    end = (j * sh) + kh
                    stw = k * sw
                    endw = (k * sw) + kw
                    X = ima[st:end, stw:endw, l]
                    if mode == "max":
                        msk = np.where(X == np.max(X), 1, 0)
                        conv[i, st:end, stw:endw, l] += msk * dA[i, j, k, l]
                    if mode == "avg":
                        av = dA[i, j, k, l] / (kh * kw)
                        msk = np.ones(X.shape)
                        conv[i, st:end, stw:endw, l] += msk * av
    return conv
