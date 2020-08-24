#!/usr/bin/env python3
""" creates a confusion matrix """

import numpy as np


def create_confusion_matrix(labels, logits):
    """
        labels is a one-hot numpy.ndarray of shape (m, classes)
        m is the number of data points
        classes is the number of classes
        logits is a one-hot numpy.ndarray of shape (m, classes)
        Returns: a confusion numpy.ndarray of shape (classes, classes)
    """
    m = labels.shape[1]
    confusion = np.array(np.zeros(m ** 2))
    confusion = confusion.reshape(m, m)
    now = np.argmax(labels, axis=1)
    pred = np.argmax(logits, axis=1)
    np.add.at(confusion, (now, pred), 1)
    return confusion
