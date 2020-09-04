#!/usr/bin/env python3
"""  performs a valid convolution on grayscale images """

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        kh is the height of the kernel
        kw is the width of the kernel
        numpy.ndarray containing the convolved images
    """
    k_h, k_w = kernel.shape
    ph, pw = padding
    newi = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    m, ih, iw = images.shape
    nh = (ih + (2 * ph) - k_h + 1)
    nw = (iw + (2 * pw) - k_w + 1)
    conv = np.zeros((m, nh, nw))
    for i in range(nh):
        for j in range(nw):
            imp = newi[:, i:i + k_h, j:j + k_w]
            res = imp * kernel
            res = res.sum(axis=1)
            res = res.sum(axis=1)
            conv[:, i, j] = res
    return conv
