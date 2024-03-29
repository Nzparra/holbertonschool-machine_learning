#!/usr/bin/env python3
""" returns two placeholders, x and y, for the neural network """
import tensorflow as tf


def create_placeholders(nx, classes):
    """
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
        Returns: placeholders named x and y, respectively
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return (x, y)
