#!/usr/bin/env python3
"""
https://www.tensorflow.org/tutorials/text/transformer#encoder
Decoder Block
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """ Encoder class """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        * N - the number of blocks in the encoder
        * dm - the dimensionality of the model
        * h - the number of heads
        * hidden - the number of hidden units in the
          fully connected layer
        * input_vocab - the size of the input vocabulary
        * max_seq_len - the maximum sequence length possible
        * drop_rate - the dropout rate
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Returns: a tensor of shape (batch, input_seq_len, dm) containing
        the encoder output
        """
        seq_len = x.shape[1]
        embedding = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedding += self.positional_encoding[:seq_len]
        encoder_out = self.dropout(embedding, training=training)
        for i in range(self.N):
            encoder_out = self.blocks[i](encoder_out, training, mask)
        return encoder_out
