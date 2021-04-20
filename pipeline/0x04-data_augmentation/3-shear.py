#!/usr/bin/env python3
""" Random Shear 3D image """
import tensorflow as tf


def shear_image(image, intensity):
    """
    image is a image to shear
    intensity should be sheared
    Returns the sheared image
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    sheared = tf.keras.preprocessing.image.random_shear(img,
                                                        intensity,
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)
    return tf.keras.preprocessing.image.array_to_img(sheared)
