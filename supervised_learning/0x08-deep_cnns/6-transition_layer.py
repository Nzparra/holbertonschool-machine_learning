#!/usr/bin/env python3
""" builds an identity block as described"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Returns: the activated output of the identity block
    """
    lay_init = K.initializers.he_normal()
    filter = int(nb_filters * compression)
    norm = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation('relu')(norm)
    conv = K.layers.Conv2D(filters=filter,
                           kernel_size=(1, 1),
                           padding="same", strides=(1, 1),
                           kernel_initializer=lay_init)(act)
    return (K.layers.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv), filter)
