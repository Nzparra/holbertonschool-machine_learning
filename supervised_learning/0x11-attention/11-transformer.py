#!/usr/bin/env python3
"""
Based on Transformer of:
https://www.tensorflow.org/tutorials/text/transformer#create_the_transformer
Transformer
"""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """ class Transform """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Sets the following public instance attributes:
        * encoder - the encoder layer
        * decoder - the decoder layer
        * linear - a final Dense layer with target_vocab units
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Returns: a tensor of shape (batch, target_seq_len, target_vocab)
        containing the transformer output
        """
        enc_output = self.encoder(inputs, training, encoder_mask)

        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)
        return final_output
