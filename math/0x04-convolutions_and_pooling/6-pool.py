#!/usr/bin/env python3
"""  performs a valid convolution on grayscale images """

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        kh is the height of the kernel
        kw is the width of the kernel
        numpy.ndarray containing the convolved images
    """
    k_h, k_w = kernel_shape
    sh, sw = stride
    m, ih, iw, ic = images.shape
    nh = int(((ih - k_h) / sh) + 1)
    nw = int(((iw - k_w) / sw) + 1)
    conv = np.zeros((m, nh, nw, ic))
    for i in range(nh):
        for j in range(nw):
            imp = images[:, (i * sh):(i * sh) + k_h, (j * sw):(sw * j) + k_w]
            if mode == 'max':
                res = np.max(imp, axis=1)
                res = np.max(res, axis=1)
            if mode == 'avg':
                res = np.mean(imp, axis=1)
                res = np.mean(res, axis=1)
            conv[:, i, j] = res
    return conv
