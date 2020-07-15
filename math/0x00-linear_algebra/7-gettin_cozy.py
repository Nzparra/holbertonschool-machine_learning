#!/usr/bin/env python3
""" concatenates two matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """
        concatenates two matrices
    """
    matrix = []
    if axis != 0:
        if (len(mat1) - len(mat2) != 0):
            return None
        for i in range(len(mat1)):
            matrix.append(mat1[i] + mat2[i])
    else:
        for i in mat1 + mat2:
            if (len(i) - len(mat1[0]) != 0):
                return None
            matrix.append(i[:])
    return matrix
