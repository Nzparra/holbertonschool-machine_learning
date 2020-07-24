#!/usr/bin/env python3
""" Calculares summation_i_squared """


def summation_i_squared(n):
    """
    n is the stopping condition
    Return the integer value of the sum
    If n is not a valid number, return None
    """
    return sum(list(map(lambda x: x**2, list(range(1, n+1))))) \
        if (n > 0) else None
