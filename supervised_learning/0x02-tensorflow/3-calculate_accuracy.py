#!/usr/bin/env python3
""" returns layer """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network’s predictions
    """
    comp = tf.math.equal(tf.math.argmax(y, axis=1),
                         tf.math.argmax(y_pred, axis=1))
    return tf.math.reduce_mean(tf.cast(comp, dtype=tf.float32))
