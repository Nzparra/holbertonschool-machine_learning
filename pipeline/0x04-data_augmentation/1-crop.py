#!/usr/bin/env python3
""" Crop a 3D image """
import tensorflow as tf


def crop_image(image, size):
    """
    * image is a 3D image to crop
    * size is a tuple
    Returns the cropped image
    """
    return tf.random_crop(value=image, size=size)
