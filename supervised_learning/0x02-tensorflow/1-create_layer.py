#!/usr/bin/env python3
""" returns layer """
import tensorflow as tf


def create_layer(prev, n, activation):
    """
                Returns: placeholders named x and y, respectively
    """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, name='layer', activacion=activation,
                            kernel_initializer=kernel)
    return layer(prev)
