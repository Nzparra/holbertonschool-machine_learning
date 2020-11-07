#!/usr/bin/env python3
""" Initialize """

import numpy as np


def pdf(X, m, S):
    """
        Returns: P, or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != S.shape[1] or S.shape[0] != S.shape[1]:
        return None
    if X.shape[1] != m.shape[0]:
        return None
    _, d = X.shape
    det = np.linalg.det(S)
    first = np.matmul((X - m), np.linalg.inv(S))
    second = np.sum(first * (X - m), axis=1)
    num = np.exp(second / -2)
    den = np.sqrt(det) * ((2 * np.pi) ** (d/2))
    pdf = num / den
    pdf = np.where(pdf < 1e-300, 1e-300, pdf)
    return pdf
