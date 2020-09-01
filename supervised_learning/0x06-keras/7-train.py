#!/usr/bin/env python3
""" builds a neural network with the Keras library """

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
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

    def learning(epoch):
        """
            Learning Rate
        """
        return (alpha / (1 + decay_rate * (epoch / 1)))
    call = None
    if validation_data:
        call.append(K.callbacks.EarlyStopping(patience=patience,
                                              monitor='val_loss'))
    if learning_rate_decay:
        call.append(K.callbacks.LearningRateSchedule(schedule=learning,
                                                     verbose=1))
    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          verbose=verbose, shuffle=shuffle, epochs=epochs,
                          validation_data=validation_data, callbacks=call)
    return history
