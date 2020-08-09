#!/usr/bin/env python3
""" Neuron that defines a single neuron performing binary classification """

import numpy as np


class Neuron():
    """
        W: The weights vector for the neuron. Upon instantiation,
        it should be initialized using a random normal distribution.
        b: The bias for the neuron. Upon instantiation,
        it should be initialized to 0.
        A: The activated output of the neuron (prediction).
        Upon instantiation, it should be initialized to 0.
    """
    def __init__(self, nx):
        """
            nx is the number of input features to the neuron
            If nx is not an integer,
            raise a TypeError with the exception: nx must be an integer
            If nx is less than 1, raise a ValueError with the exception:
            nx must be a positive integer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        self.__b = 0
        self.__A = 0
        self.__W = np.array([np.random.randn(nx)])

        @property
        def W(self):
            """  The weights vector for the neuron """
            return self.__W

        @property
        def b(self):
            """ The bias for the neuron """
            return self.__b

        @property
        def A(self):
            """ The bias for the neuron """
            return self.__A
