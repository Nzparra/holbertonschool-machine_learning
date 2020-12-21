#!/usr/bin/env python3
""" Word2vec keras """


def gensim_to_keras(model):
    """ gensim - keras """
    return model.wv.get_keras_embedding(train_embeddings=False)
