#!/usr/bin/env python3
""" calculates the shape of a matrix """


def matrix_shape(matrix):
    """ calculates the shape of a matrix """
    if not isinstance(matrix, list):
        return [0]
    if (len(matrix) > 0 and not isinstance(matrix[0], list)):
        return [len(matrix)]
    for sub_matrix in matrix:
        sub = matrix_shape(sub_matrix)
    return [len(matrix)] + sub
