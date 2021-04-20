#!/usr/bin/env python3
""" Change Brightness of a image """
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    * image is a image to change
    * max_delta is the maximum amount the image
      should be brightened
    Returns the altered image
    """
    return tf.image.adjust_brightness(image, max_delta)
