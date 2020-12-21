#!/usr/bin/env python3
""" word2vec model """

from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0,
                   workers=1):
    """ Returns: the trained model """
    if cbow is True:
        sg = 0
    else:
        sg = 1
    model = Word2Vec(size=size, window=window,
                     min_count=min_count, workers=workers,
                     sg=sg, negative=negative, seed=seed)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=iterations)
    return model
