#!/usr/bin/env python3
"""contains the NST class"""

import numpy as np
import tensorflow as tf


class NST:
    """
    NST class performs tasks for neural style transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        constructor
        :param style_image:  image used as a style reference,
            stored as a numpy.ndarray
        :param content_image: image used as a content reference,
            stored as a numpy.ndarray
        :param alpha:
        :param beta: weight for style cost
        """
        tf.enable_eager_execution()
        if type(style_image) is not np.ndarray \
                or len(style_image.shape) != 3 \
                or style_image.shape[2] != 3:
            msg = 'style_image must be a numpy.ndarray with shape (h, w, 3)'
            raise TypeError(msg)

        if type(content_image) is not np.ndarray \
                or len(content_image.shape) != 3 \
                or content_image.shape[2] != 3:
            msg = 'content_image must be a numpy.ndarray with shape (h, w, 3)'
            raise TypeError(msg)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            msg = 'alpha must be a non-negative number'
            raise TypeError(msg)

        if not isinstance(beta, (int, float)) or beta < 0:
            msg = 'beta must be a non-negative number'
            raise TypeError(msg)

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        :param image: numpy.ndarray of shape (h, w, 3)
            containing the image to be scaled
        :return:
        """
        if type(image) is not np.ndarray \
                or len(image.shape) != 3 \
                or image.shape[2] != 3:
            msg = 'image must be a numpy.ndarray with shape (h, w, 3)'
            raise TypeError(msg)
        h, w, c = image.shape
        if w > h:
            w_new = 512
            h_new = int(h * 512 / w)
        else:
            h_new = 512
            w_new = int(w * 512 / h)
        dim = (h_new, w_new)
        image = image[tf.newaxis, ...]
        image = tf.image.resize_bicubic(image, dim, align_corners=False)
        image = tf.math.divide(image, 255)
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        return image
