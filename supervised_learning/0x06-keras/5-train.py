#!/usr/bin/env python3
""" builds a neural network with the Keras library """

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
        network is the model to train
        data is a numpy.ndarray of shape (m, nx) containing the input data
        labels is a one-hot numpy.ndarray of shape (m, classes)
        batch_size is the size of the batch used for mini-batch gradient
        epochs is the number of passes through data for mini-batch gradient
        verbose is a boolean that determines if output should
        be printed during training
        shuffle is a boolean that determines whether to shuffle
        the batches every epoch.
        Returns: the History object generated after training the model
    """
    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data)
    return history
