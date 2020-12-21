#!/usr/bin/env python3
""" Bag of Words """

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ Returns: embeddings, features """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = X.toarray()
    return (embeddings, features)
