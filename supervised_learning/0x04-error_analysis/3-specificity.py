#!/usr/bin/env python3
""" calculates the sensitivity for each class in a confusion matrix """


def specificity(confusion):
    """
        confusion is a confusion numpy.ndarray of shape (classes, classes)
        classes is the number of classes
        Returns: a numpy.ndarray of shape (classes,)
    """
    tp = confusion.diagonal()
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    tn = confusion.sum() - (tp + fp + fn)
    return (tn / (tn + fp))
