#!/usr/bin/env python3
""" TF-IDF """

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ Returns: embeddings, features """
    tf_idf = TfidfVectorizer(vocabulary=vocab)
    X = tf_idf.fit_transform(sentences)
    features = tf_idf.get_feature_names()
    embeddings = X.toarray()
    return (embeddings, features)
