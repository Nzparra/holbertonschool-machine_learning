#!/usr/bin/env python3
""" BIC            """

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
        Returns: best_k, best_result, l, b, or None
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None, None)
    if type(kmin) is not int or kmin <= 0:
        return (None, None, None, None)
    if kmax is not None and type(kmax) is not int:
        return (None, None, None, None)
    if kmax is None:
        kmax = X.shape[0]
    if kmin >= kmax or kmax <= 0:
        return (None, None, None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None, None, None)
    if type(tol) is not float or tol < 0:
        return (None, None, None, None)
    if type(verbose) is not bool:
        return (None, None, None, None)
    n, d = X.shape
    all_l = np.zeros(kmax - kmin + 1)
    all_BIC = np.zeros(kmax - kmin + 1)
    count = 0
    all_pi = []
    all_S = []
    all_m = []
    for i in range(kmin, kmax + 1):
        pi, m, S, _, log_like = expectation_maximization(X, i, iterations,
                                                         tol, verbose)
        all_pi.append(pi)
        all_m.append(m)
        all_S.append(S)
        all_l[count] = log_like
        p = (i - 1) + (i * d) + ((i * d) * (d + 1) / 2)
        BIC = (p) * np.log(n) - 2 * log_like
        all_BIC[count] = BIC
        count = count + 1
    min_BIC = np.argmin(all_BIC)
    best_pi = all_pi[min_BIC]
    best_m = all_m[min_BIC]
    best_S = all_S[min_BIC]
    best_k = kmin + min_BIC
    return [best_k, (best_pi, best_m, best_S), all_l, all_BIC]
