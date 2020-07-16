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


def add(mat1, mat2):
    """ Auxiliar to add """
    if type(mat1[0]) is not list:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        matrix = []
        for j in range(len(mat1)):
            matrix.append(add(mat1[j], mat2[j]))
        return matrix


def add_matrices(mat1, mat2):
    """ adds two matrices """
    if (matrix_shape(mat1) != matrix_shape(mat2)):
        return None
    else:
        return add(mat1, mat2)
