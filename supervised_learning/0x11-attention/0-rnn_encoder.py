#!/usr/bin/env python3
""" RNN Encoder """

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ RNNEncoder """

    def __init__(self, vocab, embedding, units, batch):
        """
         Recurrent weights should be initialized with glorot_uniform
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
            Returns: a tensor of shape (batch, units)containing
            the initialized hidden states
        """
        initializer = tf.keras.initializers.Zeros()
        values = initializer(shape=(self.batch, self.units))
        return values

    def call(self, x, initial):
        """
            Returns: outputs, hidden
        """
        out_embedding = self.embedding(x)
        out_encoder, hidden_encoder = self.gru(inputs=out_embedding,
                                               initial_state=initial)
        return out_encoder, hidden_encoder
