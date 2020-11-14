#!/usr/bin/env python3
""" backward Algorithm """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
        Returns: P, B, or None, None on failure
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    for elem in Observation:
        if elem < 0:
            return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    sum1 = np.sum(Emission, axis=1)
    if not (sum1 == 1.).all():
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    sum2 = np.sum(Transition, axis=1)
    if not (sum2 == 1.).all():
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    sum3 = np.sum(Initial)
    if sum3 != 1.:
        return None, None
    if Initial.shape[0] != Transition.shape[0]:
        return None, None
    if Emission.shape[0] != Transition.shape[0]:
        return None, None
    N, M = Emission.shape
    T = Observation.shape[0]
    beta = np.zeros((N, T))
    beta[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        for n in range(N):
            trans = Transition[n]
            em = Emission[:, Observation[t + 1]]
            post = beta[:, t + 1]
            beta[n, t] = np.sum(trans * post * em)
    prob = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])
    return (prob, beta)
