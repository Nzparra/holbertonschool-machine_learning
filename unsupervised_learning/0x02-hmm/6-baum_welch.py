#!/usr/bin/env python3
""" Baum-Welch """

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    forward hidden Markov model
    """
    N, M = Emission.shape
    T = Observation.shape[0]
    alpha = np.zeros((N, T))
    aux = (Initial.T * Emission[:, Observation[0]])
    alpha[:, 0] = aux.reshape(-1)
    for t in range(1, T):
        prev = alpha[:, t - 1]
        trans = Transition
        em = Emission[:, Observation[t]]
        first = np.matmul(prev, trans)
        alpha[:, t] = first * em
    return (alpha)


def backward(Observation, Emission, Transition, Initial):
    """
    backward hidden Markov model
    """
    N, M = Emission.shape
    T = Observation.shape[0]
    beta = np.zeros((N, T))
    beta[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        trans = Transition
        em = Emission[:, Observation[t + 1]]
        post = beta[:, t + 1]
        first = post * em
        beta[:, t] = np.dot(trans, first)
    return (beta)


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
        Returns: the converged Transition, Emission, or None
    """
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    for elem in Observations:
        if elem < 0:
            return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if Initial.shape[0] != Transition.shape[0]:
        return None, None
    if Emission.shape[0] != Transition.shape[0]:
        return None, None
    N, M = Emission.shape
    T = Observations.shape[0]
    Obs = Observations.copy()
    Init = Initial.copy()
    Emi = Emission.copy()
    Trans = Transition.copy()
    for n in range(iterations):
        alpha = forward(Obs, Emi, Trans, Init)
        beta = backward(Obs, Emi, Trans, Init)
        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            first = np.dot(alpha[:, t].T, Trans)
            second = Emi[:, Obs[t + 1]].T
            denominator = np.dot(first * second, beta[:, t + 1])
            for i in range(N):
                first = alpha[i, t] * Trans[i]
                second = Emi[:, Obs[t + 1]].T
                numerator = first * second * beta[:, t + 1].T
                xi[i, :, t] = numerator / denominator
        gamma = np.sum(xi, axis=1)
        Trans = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                         axis=0).reshape((-1, 1))))
        denominator = np.sum(gamma, axis=1)
        for l in range(M):
            Emi[:, l] = np.sum(gamma[:, Obs == l], axis=1)
        Emi = np.divide(Emi, denominator.reshape((-1, 1)))
    return (Trans, Emi)
