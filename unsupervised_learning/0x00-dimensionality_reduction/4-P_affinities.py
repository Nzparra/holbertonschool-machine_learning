#!/usr/bin/env python3
"""
Stochastic Neightbor Embedding
"""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Returns: P, a numpy.ndarray of shape (n, n) containing the symmetric
    P affinities
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)
    for i in range(n):
        copy = D[i].copy()
        copy = np.delete(copy, i, axis=0)
        Hi, Pi = HP(copy, betas[i])
        Hdiff = Hi - H
        betamin = None
        betamax = None
        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                betamin = betas[i, 0]
                if betamax is None:
                    betas[i, 0] = betas[i, 0] * 2
                else:
                    betas[i, 0] = (betas[i, 0] + betamax) / 2
            else:
                betamax = betas[i, 0]
                if betamin is None:
                    betas[i, 0] = betas[i, 0] / 2
                else:
                    betas[i, 0] = (betas[i, 0] + betamin) / 2
            Hi, Pi = HP(copy, betas[i])
            Hdiff = Hi - H
        Pi = np.insert(Pi, i, 0)
        P[i] = Pi
    sym_P = (P.T + P) / (2 * n)
    return sym_P
