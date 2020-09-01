#!/usr/bin/env python3
""" builds a neural network with the K library """

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
        The last dimension of the one-hot matrix must be the number of classes
        Returns: the one-hot matrix
    """
    return K.utils.to_categorical(y=labels, num_classes=classes)
