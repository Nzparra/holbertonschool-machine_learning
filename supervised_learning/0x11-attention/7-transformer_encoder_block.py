#!/usr/bin/env python3
"""
based on encoderlayer of:
https://www.tensorflow.org/tutorials/text/transformer#encoder_layer
Encoder Block
remember to use tensorflow 1.15 or higher
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    class Encoder block
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        * dropout1 - the first dropout layer
        * dropout2 - the second dropout layer
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Returns: a tensor of shape (batch, input_seq_len, dm) containing
        the block’s output
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        hidden_output = self.dense_hidden(out1)
        output_output = self.dense_output(hidden_output)
        ffn_output = self.dropout2(output_output, training=training)
        final_output = self.layernorm2(out1 + ffn_output)
        return final_output
