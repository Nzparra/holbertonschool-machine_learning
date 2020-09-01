#!/usr/bin/env python3
""" saves an entire model """

import tensorflow.keras as K


def save_config(network, filename):
    """
        network is the model to save
        filename is the path of the file that the model should be saved to
        Returns: None
    """
    with open(filename, 'w') as file:
        json_m = network.to_json()
        file.write(json_m)
    return None


def load_config(filename):
    """
        network is the model to which the weights should be loaded
        filename path of the file that the weights should be loaded from
        Returns: None
    """
    with open(filename, 'r') as file:
        json_m = file.read()
        model = K.models.model_from_json(json_m)
    return model
