#!/usr/bin/env python3
""" defines a Deep neural network """

import numpy as np


class DeepNeuralNetwork():
    """
    L: The number of layers in the neural network.
    cache: A dictionary to hold all intermediary values of the network.
    weights: A dictionary to hold all weights and biased of the network.
    """

    def __init__(self, nx, layers):
        """ Init Method to DeepNeural"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i, layer in enumerate(layers):
            if type(layer) is not int or layer < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                value = np.random.randn(layer, nx) * np.sqrt(2/nx)
                self.weights['W' + str(i + 1)] = value
            else:
                value2 = np.random.randn(layer, layers[i - 1])
                value3 = np.sqrt(2 / layers[i - 1])
                self.weights['W' + str(i + 1)] = value2 * value3
            self.weights['b' + str(i + 1)] = np.zeros((layer, 1))

    @property
    def L(self):
        """ L attribute """
        return self.__L

    @property
    def cache(self):
        """ cache attribute """
        return self.__cache

    @property
    def weights(self):
        """ weights attribute """
        return self.__weights
