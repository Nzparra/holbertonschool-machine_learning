#!/usr/bin/env python3
""" normalizes (standardizes) a matrix """


def normalize(X, m, s):
    """
        m is the number of data points
        nx is the number of features
    """
    return ((X - m) / s)
