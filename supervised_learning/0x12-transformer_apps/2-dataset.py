#!/usr/bin/env python3
"""
based on:
https://www.tensorflow.org/tutorials/text/transformer
Creating Class Dataset
"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """
    Class Dataset
    """

    def __init__(self):
        """
        tokenizer_en is the English tokenizer created from the
        training set
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        PT, EN = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = PT, EN

    def tokenize_dataset(self, data):
        """
        Returns: tokenizer_pt, tokenizer_en
        """
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        return (tokenizer_pt, tokenizer_en)

    def encode(self, pt, en):
        """
        Returns: pt_tokens, en_tokens
        """
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return lang1, lang2

    def tf_encode(self, pt, en):
        """ return tensors """
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
