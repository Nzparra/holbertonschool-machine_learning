#!/usr/bin/env python3
""" F1  Score  Test py"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
        confusion is a confusion numpy.ndarray of shape (classes, classes)
        classes is the number of classes
        Returns: a numpy.ndarray of shape (classes,)
    """
    s = sensitivity(confusion)
    p = precision(confusion)
    return (2 * s * p) / (s + p)
