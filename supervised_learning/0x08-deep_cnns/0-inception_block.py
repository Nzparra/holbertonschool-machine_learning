#!/usr/bin/env python3
""" builds an inception block as described in Going Deeper with Convolutions"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
        Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    lay_init = K.initializers.he_normal()
    convF1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                             activation='relu', kernel_initializer=lay_init)
    convF3R = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), padding='same',
                              activation='relu', kernel_initializer=lay_init)
    convF3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                             activation='relu', kernel_initializer=lay_init)
    convF5R = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding='same',
                              activation='relu', kernel_initializer=lay_init)
    convF5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                             activation='relu', kernel_initializer=lay_init)
    maxpool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                    padding='same')
    convFPP = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), padding='same',
                              activation='relu', kernel_initializer=lay_init)
    xF1 = convF1(A_prev)
    xF3 = convF3(convF3R(A_prev))
    xF5 = convF5(convF5R(A_prev))
    xFPP = convFPP(maxpool(A_prev))
    return K.layers.concatenate([xF1, xF3, xF5, xFPP])
