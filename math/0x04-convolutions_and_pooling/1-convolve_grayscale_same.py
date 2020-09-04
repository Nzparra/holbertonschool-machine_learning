#!/usr/bin/env python3
"""  performs a valid convolution on grayscale images """

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        kh is the height of the kernel
        kw is the width of the kernel
        numpy.ndarray containing the convolved images
    """
    k_h, k_w = kernel.shape
    if k_h % 2 == 0:
        ph = int(k_h / 2)
    else:
        ph = int((k_h - 1) / 2)
    if k_w % 2 == 0:
        pw = int(k_w / 2)
    else:
        pw = int((k_w - 1) / 2)
    new_i = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    m, ih, iw = images.shape
    conv = np.zeros((m, ih, iw))
    for i in range(ih):
        for j in range(iw):
            imp = new_i[:, i:i + k_h, j:j + k_w]
            res = imp * kernel
            res = res.sum(axis=1)
            res = res.sum(axis=1)
            conv[:, i, j] = res
    return conv
