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
    in_ = K.layers.Input(shape=(nx,))
    out = K.layers.Dense(layers[0],
                         activation=activations[0],
                         kernel_regularizer=kernel)(in_)
    for i in range(1, len(layers)):
        dout = K.layers.Dropout(1-keep_prob)(out)
        out = K.layers.Dense(layers[i],
                             activation=activations[i],
                             kernel_regularizer=kernel)(dout)
    kmodel = K.models.Model(inputs=in_, outputs=out)
    return kmodel
