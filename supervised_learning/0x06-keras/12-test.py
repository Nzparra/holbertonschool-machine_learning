#!/usr/bin/env python3
""" saves an entire model """

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
        network is the network model to test
        data is the input data to test the model with
        labels are the correct one-hot labels of data
        verbose is a boolean that determines
        Returns: the loss and accuracy of the model with the testing data
    """
    loss = network.evaluate(x=data, y=labels, verbose=verbose)
    return loss
