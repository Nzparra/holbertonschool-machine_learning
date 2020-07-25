#!/usr/bin/env python3
""" Derivate """


def poly_derivative(poly):
    """
    Derivate a polynomial
    """
    if(len(poly) == 0 or type(poly) != list):
        return None
    for i in poly:
        if(type(i) != int and type(i) != float):
            return None
    if(len(poly) == 1):
        return [0]
    derivate = []
    for j in range(1, len(poly)):
        derivate.append(poly[j] * j)
    return derivate
