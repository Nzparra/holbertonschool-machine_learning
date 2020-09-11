#!/usr/bin/env python3
"""  performs forward propagation over a
    convolutional layer of a neural network
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
        Returns: the output of the convolutional layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    sh, sw = stride
    kh, kw, c_prev, c_new = W.shape
    if padding == "valid":
        ph, pw = (0, 0)
    if padding == "same":
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    A_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dA_pad = np.pad(dA, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    for i in range(m):
        ima = A_pad[i]
        Dima = dA_pad[i]
        for j in range(h_new):
            for k in range(w_new):
                for l in range(c_new):
                    st = j * sh
                    end = (j * sh) + kh
                    stw = k * sw
                    endw = (k * sw) + kw
                    x = ima[st:end, stw:endw]
                    aux = W[:, :, :, l] * dZ[i, j, k, l]
                    Dima[st:end, stw:endw] += aux
                    dW[:, :, :, l] += x * dZ[i, j, k, l]
        if (padding == 'valid'):
            dA[i] += Dima
        if (padding == 'same'):
            dA[i] += Dima[ph: -ph, pw: -pw]
    return (dA, dW, db)
