#!/usr/bin/env python3
"""  performs a valid convolution on grayscale images """

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        kh is the height of the kernel
        kw is the width of the kernel
        numpy.ndarray containing the convolved images
    """
    k_h, k_w, k_c = kernel.shape
    sh, sw = stride
    m, ih, iw, ic = images.shape
    ph, pw = (0, 0)
    if padding == 'same':
        ph = int((((ih - 1) * sh + k_h - ih) / 2) + 1)
        pw = int((((iw - 1) * sw + k_w - iw) / 2) + 1)
    if type(padding) is tuple and len(padding) == 2:
        ph, pw = padding
    newi = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    nh = (((ih + (2 * ph) - k_h) // sh) + 1)
    nw = (((iw + (2 * pw) - k_w) // sw) + 1)
    conv = np.zeros((m, nh, nw))
    for i in range(0, nh):
        for j in range(0, nw):
            imp = newi[:, (i * sh):(i * sh) + k_h, (j * sw):(sw * j) + k_w]
            res = imp * kernel
            res = res.sum(axis=1)
            res = res.sum(axis=1)
            res = res.sum(axis=1)
            conv[:, i, j] = res
    return conv
