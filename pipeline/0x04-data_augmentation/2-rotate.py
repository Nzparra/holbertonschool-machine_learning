#!/usr/bin/env python3
""" Rotate 90 degrees a 3d """
import tensorflow as tf


def rotate_image(image):
    """
    image is a image to rotate
    Returns the rotated image
    """
    return tf.image.rot90(image=image, k=1, name=None)
