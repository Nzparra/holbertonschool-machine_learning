#!/usr/bin/env python3
""" returns layer """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network’s predictions
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
