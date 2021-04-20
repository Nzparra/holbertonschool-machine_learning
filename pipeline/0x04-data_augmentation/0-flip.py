#!/usr/bin/env python3
""" Flip an 3D image tf"""
import tensorflow as tf


def flip_image(image):
    """
    * image is a 3D tf.Tensor containing the image to flip
    * Returns image
    """
    return tf.image.flip_left_right(image)
