#!/usr/bin/env python3
""" Viretbi """

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
        Returns: path, P, or None, None on failure
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
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))
    aux = (Initial.T * Emission[:, Observation[0]])
    viterbi[:, 0] = aux.reshape(-1)
    backpointer[:, 0] = 0
    for t in range(1, T):
        for n in range(N):
            prev = viterbi[:, t - 1]
            trans = Transition[:, n]
            em = Emission[n, Observation[t]]
            result = prev * trans * em
            viterbi[n, t] = np.amax(result)
            backpointer[n, t - 1] = np.argmax(result)
    path = []
    last_state = np.argmax(viterbi[:, T - 1])
    path.append(int(last_state))
    for i in range(T - 2, -1, -1):
        path.append(int(backpointer[int(last_state), i]))
        last_state = backpointer[int(last_state), i]
    path.reverse()
    min_prob = np.amax(viterbi, axis=0)
    min_prob = np.amin(min_prob)
    return (path, min_prob)
