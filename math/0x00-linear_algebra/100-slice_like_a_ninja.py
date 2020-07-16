#!/usr/bin/env python3
""" slices a matrix along a specific axes """


def np_slice(matrix, axes={}):
    """ slices a matrix along a specific axes"""
    temp = matrix.ndim * [slice(None)]
    for i, j in axes.items():
        temp[i] = slice(*j)
    return matrix[tuple(temp)]
