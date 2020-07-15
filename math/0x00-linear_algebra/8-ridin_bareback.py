#!/usr/bin/env python3
""" multiply two matrices """


def mat_mul(mat1, mat2):
    """
        multiply two matrices
    """
    x = [[sum(a*b for a, b in zip(mat1_row, mat2_col))
          for mat2_col in zip(*mat2)] for mat1_row in mat1]
    return x
