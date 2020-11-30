#!/usr/bin/env python3
""" Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
        Returns: encoder, decoder, auto
    """
    X_encode = keras.Input(shape=input_dims)

    conv_e = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                 padding='same', activation='relu')(X_encode)

    pool_e = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                       padding="same")(conv_e)

    for i in range(1, len(filters)):
        conv_e = keras.layers.Conv2D(filters=filters[i],
                                     kernel_size=(3, 3),
                                     padding='same',
                                     activation='relu')(pool_e)
        pool_e = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           padding="same")(conv_e)
    latent_e = pool_e
    encoder = keras.Model(inputs=X_encode, outputs=latent_e)
    X_decode = keras.Input(shape=latent_dims)
    conv_d = keras.layers.Conv2D(filters=filters[-1], kernel_size=(3, 3),
                                 padding='same', activation='relu')(X_decode)
    pool_d = keras.layers.UpSampling2D((2, 2))(conv_d)
    for j in range(len(filters) - 2, 0, -1):
        conv_d = keras.layers.Conv2D(filters=filters[j], kernel_size=(3, 3),
                                     padding='same', activation='relu')(pool_d)
        pool_d = keras.layers.UpSampling2D((2, 2))(conv_d)
    conv_d = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                 padding='valid', activation='relu')(pool_d)
    pool_d = keras.layers.UpSampling2D((2, 2))(conv_d)
    output = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                                 padding='same', activation='sigmoid')(pool_d)
    decoder = keras.Model(inputs=X_decode, outputs=output)
    X_input = keras.Input(shape=input_dims)
    encode_output = encoder(X_input)
    decoder_output = decoder(encode_output)
    auto = keras.Model(inputs=X_input, outputs=decoder_output)
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return (encoder, decoder, auto)
