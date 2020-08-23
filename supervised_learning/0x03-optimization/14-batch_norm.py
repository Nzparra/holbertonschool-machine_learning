#!/usr/bin/env python3
""" batch normalization layer for a neural network in tensorflow """

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
        prev is the activated output of the previous layer
        n is the number of nodes in the layer to be created
        activation is the activatio
        Returns: a tensor of the activated output for the layer
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lay = tf.layers.Dense(units=n, kernel_initializer=w)
    Z = lay(prev)
    gamma = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[n]),
                        name='gamma', trainable=True)
    beta = tf.Variale(tf.constant(0, dtype=tf.float32, shape=[n]),
                      name='beta', trainable=True)
    eps = tf.constant(1e-8)
    m, v = tf.nn.moments(Z, axes=[0])
    Z_ = tf.nn.batch_normalization(x=Z, mean=m, variance=v,
                                   offset=beta, scale=gamma,
                                   variance_epsilon=eps)
    if activation:
        return activation(Z_)
    else:
        return Z_
