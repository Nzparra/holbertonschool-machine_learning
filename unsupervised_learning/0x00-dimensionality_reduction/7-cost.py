#!/usr/bin/env python3
""" Stochastic Neighbor Embedding """

import numpy as np


def cost(P, Q):
    """ Returns: C, the cost of the transformation """
    Q_new = np.where(Q == 0, 1e-12, Q)
    P_new = np.where(P == 0, 1e-12, P)
    C = np.sum(P * np.log(P_new / Q_new))
    return C
