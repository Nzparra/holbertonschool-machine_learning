#!/usr/bin/env python3
""" change Hue of image"""
import tensorflow as tf


def change_hue(image, delta):
    """
    image is a image to change
    delta is the amount the hue should change
    Returns the altered image
    """
    return tf.image.adjust_hue(image, delta)
