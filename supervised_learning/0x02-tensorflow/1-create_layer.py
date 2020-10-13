#!/usr/bin/env python3
""" returns layer """
import tensorflow as tf


def create_layer(prev, n, activation):
    """
                Returns: placeholders named x and y, respectively
    """
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=k_init,
        activation=activation,
        name='Layer'
    )
    y = layer(prev)
    return y
