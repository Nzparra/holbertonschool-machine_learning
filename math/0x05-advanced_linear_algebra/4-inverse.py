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


def minor(matrix):
    """ Returns: the minor matrix of matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for elem in matrix:
        if type(elem) is not list:
            raise TypeError('matrix must be a list of lists')
    size = len(matrix)
    for elem in matrix:
        if len(elem) != size or len(elem) == 0:
            raise ValueError('matrix must be a non-empty square matrix')
    if size == 1:
        return [[1]]
    else:
        minor = []
        for i in range(size):
            row = []
            count = 0
            while (count < size):
                copy = []
                for elem in matrix:
                    copy.append(list(elem))
                copy.pop(i)
                new_mat = []
                for elem in copy:
                    elem.pop(count)
                    new_mat.append(elem)
                row.append(determinant(new_mat))
                count += 1
            minor.append(row)
        return minor


def cofactor(matrix):
    """ cofactor matrix of matrix """
    min_matrix = minor(matrix)
    for i in range(len(min_matrix)):
        for j in range(len(min_matrix[i])):
            if (i + j) % 2 != 0:
                min_matrix[i][j] = -1 * min_matrix[i][j]
    return min_matrix


def adjugate(matrix):
    """ Returns: the adjugate matrix of matrix """
    cof_mat = cofactor(matrix)
    adj = []
    for i in range(len(cof_mat)):
        row = []
        for j in range(len(cof_mat)):
            row.append(cof_mat[j][i])
        adj.append(row)
    return adj


def inverse(matrix):
    """ Returns: the inverse of matrix """
    adj_mat = adjugate(matrix)
    det = determinant(matrix)
    if det == 0:
        return None
    inverse = []
    for elem in adj_mat:
        min_inv = [number/det for number in elem]
        inverse.append(min_inv)
    return inverse
