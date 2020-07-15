#!/usr/bin/env python3
""" Traspose a 2D matrix """


def matrix_transpose(matrix):
    """ Traspose a 2D matrix """
    transpose = []
    for row in map(list, zip(*matrix)):
        transpose.append(row)
    return transpose
