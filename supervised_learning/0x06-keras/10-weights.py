#!/usr/bin/env python3
""" saves an entire model """

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
        network is the model to save
        filename is the path of the file that the model should be saved to
        Returns: None
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
        network is the model to which the weights should be loaded
        filename path of the file that the weights should be loaded from
        Returns: None
    """
    network.load_weights(filepath=filename)
    return None
