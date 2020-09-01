#!/usr/bin/env python3
""" builds a neural network with the K library """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        nx is the number of input features to the network
        layers is a list containing the number of nodes
        activations is a list containing the activation functions
        lambtha is the L2 regularization parameter
        keep_prob is the probability that a node will be kept for dropout
        Returns: the K model
    """
    kernel = K.regularizers.l2(lambtha)
    kmodel = K.Sequential()
    kmodel.add(K.layers.Dense(units=layers[0],
                              activation=activations[0],
                              kernel_regularizer=kernel,
                              input_shape=(nx,)))
    for i in range(1, len(layers)):
        kmodel.add(K.layers.Dropout(1 - keep_prob))
        kmodel.add(K.layers.Dense(units=layers[i],
                                  activation=activations[i],
                                  kernel_regularizer=kernel))
    return kmodel
