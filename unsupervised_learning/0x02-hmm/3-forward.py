#!/usr/bin/env python3
""" Forward MC """

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
        Returns: P, F, or None, None on failure
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
    alpha = np.zeros((N, T))
    aux = (Initial.T * Emission[:, Observation[0]])
    alpha[:, 0] = aux.reshape(-1)
    for t in range(1, T):
        for n in range(N):
            prev = alpha[:, t - 1]
            trans = Transition[:, n]
            em = Emission[n, Observation[t]]
            alpha[n, t] = np.sum(prev * trans * em)
    prop = np.sum(alpha[:, -1])
    return (prop, alpha)
