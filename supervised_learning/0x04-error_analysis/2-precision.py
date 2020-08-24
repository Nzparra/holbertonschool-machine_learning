#!/usr/bin/env python3
""" calculates the sensitivity for each class in a confusion matrix """

import numpy as np


def precision(confusion):
    """
        confusion is a confusion numpy.ndarray of shape (classes, classes)
        classes is the number of classes
        Returns: a numpy.ndarray of shape (classes,)
    """
    return ((confusion.diagonal()) / (confusion.sum(axis=0)))
