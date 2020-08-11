#!/usr/bin/env python3
""" defines a Deep neural network """

import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache["A0"] = X
        for i in range(self.__L):
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            A = self.__cache['A' + str(i)]
            A = 1 / (1 + np.exp(-(np.matmul(W, A) + b)))
            self.__cache['A' + str(i + 1)] = A
        return (A, self.__cache)

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        return (-1 / Y.shape[1]) * np.sum(Y * np.log(A) + (1 - Y) *
                                          (np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        A, self.__cache = self.forward_prop(X)
        return ((np.where(A >= 0.5, 1, 0)), self.cost(Y, A))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        dz = 0
        value = {}
        for i in range(self.__L, 0, -1):
            if i == self.__L:
                dz = cache['A' + str(i)] - Y
                val = str(i - 1)
                dw = np.matmul(dz, cache['A' + val].transpose()) / Y.shape[1]
                db = (dz.sum(axis=1, keepdims=True)) / Y.shape[1]
            else:
                weight = self.__weights['W' + str(i + 1)].transpose()
                dzl = np.matmul(weight, dz) *\
                    cache['A' + str(i)] * (1 - cache['A' + str(i)])
                dw = np.matmul(dzl, cache['A' +
                                          str(i - 1)].transpose()) / Y.shape[1]
                db = (dzl.sum(axis=1, keepdims=True)) / Y.shape[1]
                dz = dzl
            value['W' + str(i)] = self.__weights['W' + str(i)] - (alpha * dw)
            value['b' + str(i)] = self.__weights['b' + str(i)] - (alpha * db)
        self.__weights = value

    def train(self, X, Y,
              iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Trains the deep neural network """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        costs = []
        for i in range(iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            costs.append(self.cost(Y, A))
            if (verbose and i % step == 0):
                print("Cost after {} iterations: {}".format(i, costs[i]))
        if graph:
            plt.plot(np.arange(0, iterations), costs)
            plt.title('Training Cost')
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Saves the instance object to a file in pickle format
        """
        if '.pkl' not in filename:
            filename = filename + '.pkl'
        with open(filename, 'wb') as archive:
            pickle.dump(self, archive)

    @staticmethod
    def load(filename):
        """
            Loads a pickled DeepNeuralNetwork object
        """
        try:
            with open(filename, 'rb') as archive:
                open_file = pickle.load(archive)
            return open_file
        except FileNotFoundError:
            return None
