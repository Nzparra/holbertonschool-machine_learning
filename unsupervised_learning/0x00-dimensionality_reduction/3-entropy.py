#!/usr/bin/env python3
""" Shannon entropy """

import numpy as np


def HP(Di, beta):
    """ Returns: (Hi, Pi) """
    numerator = np.exp(-Di.copy() * beta)
    denominator = np.sum(np.exp(-Di.copy() * beta))
    Pi = numerator / denominator
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
