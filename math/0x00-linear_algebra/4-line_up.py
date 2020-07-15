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
