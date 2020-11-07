#!/usr/bin/env python3
""" kmeans using scikit """

import sklearn.mixture


def gmm(X, k):
    """
        Returns: pi, m, S, clss, bic
    """
    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    GMM.fit(X)
    m = GMM.means_
    S = GMM.covariances_
    pi = GMM.weights_
    clss = GMM.predict(X)
    BIC = GMM.bic(X)
    return pi, m, S, clss, BIC
