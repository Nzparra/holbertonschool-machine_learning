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