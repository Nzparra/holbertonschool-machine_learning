#!/usr/bin/env python3
"""
https://towardsdatascience.com/implementing-neural-machine-
translation-with-attention-using-tensorflow-fc9c6f26155f
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ SelfAttention """

    def __init__(self, units):
        """
            V - a Dense layer with 1 units, to be applied to the
            tanh of the sum of the outputs of W and U
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
            Returns: context, weights
        """
        s_expanded = tf.expand_dims(input=s_prev, axis=1)
        first = self.W(s_expanded)
        second = self.U(hidden_states)
        score = self.V(tf.nn.tanh(first + second))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
