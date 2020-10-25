#!/usr/bin/env python3
""" calculates the determinant of a matrix """


def determinant(matrix):
    """ Returns the determinant of matrix """
    if type(matrix) is list and len(matrix) == 1:
        if type(matrix[0]) is list and len(matrix[0]) == 0:
            return 1
        if type(matrix[0]) is list and len(matrix[0]) == 1:
            return matrix[0][0]
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for elem in matrix:
        if type(elem) is not list:
            raise TypeError('matrix must be a list of lists')
    size = len(matrix)
    for elem in matrix:
        if len(elem) != size:
            raise ValueError('matrix must be a square matrix')
    if size == 2:
        det = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return det
    else:
        det = 0
        count = 0
        cof = 1
        while (count < size):
            multy = matrix[0][count]
            multy = multy * cof
            copy = []
            for elem in matrix:
                copy.append(list(elem))
            copy.pop(0)
            new_mat = []
            for elem in copy:
                elem.pop(count)
                new_mat.append(elem)
            mini_det = determinant(new_mat)
            det = det + (multy * mini_det)
            cof = cof * -1
            count += 1
        return det
