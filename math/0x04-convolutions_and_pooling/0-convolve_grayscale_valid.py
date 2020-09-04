#!/usr/bin/env python3
"""  performs a valid convolution on grayscale images """

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        kh is the height of the kernel
        kw is the width of the kernel
        numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]
    outh = images.shape[1] - k_h + 1
    outw = images.shape[2] - k_w + 1
    conv = np.zeros((m, outh, outw))
    img = np.arange(m)
    for i in range(outh):
        for j in range(outw):
            mul = images[img, i:k_h+i, j:k_w+j]
            conv[img, i, j] = np.sum(np.multiply(mul, kernel), axis=(1, 2))
    return conv
