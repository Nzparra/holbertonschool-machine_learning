#!/usr/bin/env python3
""" concatenates two matrices along a specific axis """


import numpy


def np_cat(mat1, mat2, axis=0):
    """ concatenates two matrices along a specific axis"""
    return numpy.concatenate((mat1, mat2), axis=axis)
