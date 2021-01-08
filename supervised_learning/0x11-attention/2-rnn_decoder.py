#!/usr/bin/env python3
"""
https://towardsdatascience.com/implementing-neural-machine-
translation-with-attention-using-tensorflow-fc9c6f26155f
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    class RNNDecoder
    """

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
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
            Returns: y, s
        """
        context_vector, attention_weights = self.attention(s_prev,
                                                           hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return y, state
