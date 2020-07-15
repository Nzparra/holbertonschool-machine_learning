#!/usr/bin/env python3
""" adds two matrices element-wise """


def add_arrays(arr1, arr2):
    """
        adds two matrices element-wise
    """
    if (len(arr1) != len(arr2)):
        return None
    else:
        return [An + Bn for An, Bn in zip(arr1, arr2)]


def add_matrices2D(mat1, mat2):
    """
    that adds two matrices element-wise
    """
    if (len(mat1) - len(mat2) != 0):
        return None
    else:
        matrix = [add_arrays(An, Bm) for An, Bm in zip(mat1, mat2)]
        if matrix[0] is not None:
            return matrix
