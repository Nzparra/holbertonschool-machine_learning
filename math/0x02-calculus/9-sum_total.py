#!/usr/bin/env python3


def summation_i_squared(n):
    """
    n is the stopping condition
    Return the integer value of the sum
    If n is not a valid number, return None
    """
    return sum(list(map(lambda x: x*x, list(range(1, n+1))))) \
        if (n is not None or n >= 0) else None
