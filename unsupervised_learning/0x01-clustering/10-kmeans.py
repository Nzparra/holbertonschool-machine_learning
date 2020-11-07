#!/usr/bin/env python3
""" kmeans using scikit """

import sklearn.cluster


def kmeans(X, k):
    """
        Returns: C, clss
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
