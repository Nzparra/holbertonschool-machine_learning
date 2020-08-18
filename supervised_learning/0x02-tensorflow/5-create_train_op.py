#!/usr/bin/env python3
""" returns layer """
import tensorflow as tf


def create_train_op(loss, alpha):
    """
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network’s predictions
    """
    train = tf.train.GradientDescentOptimizer(alpha)
    train = train.minimize(loss)
    return train
