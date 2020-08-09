#!/usr/bin/env python3
""" defines a neural network with one hidden layer """

import numpy as np


class NeuralNetwork():
    """
    W1: The weights vector for the hidden layer.
    b1: The bias for the hidden layer.
    A1: The activated output for the hidden layer.
    W2: The weights vector for the output neuron.
    b2: The bias for the output neuron.
    A2: The activated output for the output neuron (prediction).
    """

    def __init__(self, nx, nodes):
        """ Init"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0

    @property
    def W1(self):
        """ W1 """
        return self.__W1

    @property
    def b1(self):
        """ b1 """
        return self.__b1

    @property
    def A1(self):
        """ A1 """
        return self.__A1

    @property
    def W2(self):
        """ W2 """
        return self.__W2

    @property
    def b2(self):
        """ b2 """
        return self.__b2

    @property
    def A2(self):
        """ A2 """
        return self.__A2
