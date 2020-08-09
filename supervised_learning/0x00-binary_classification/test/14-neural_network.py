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
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

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

    def forward_prop(self, X):
        """ forward propagation of the neural network """
        self.__A1 = 1 / (1 + np.exp(-(np.matmul(self.__W1, X) + self.__b1)))
        self.__A2 = 1 / (1 + np.exp(-(np.matmul(self.__W2, self.__A1)
                                      + self.__b2)))
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """ cost of the model using logistic regression """
        return (-1 / Y.shape[1]) * np.sum(Y * np.log(A) + (1 - Y)
                                          * (np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        self.__A = self.forward_prop(X)
        return ((np.where(self.__A2 >= 0.5, 1, 0)), self.cost(Y, self.__A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        dw1 = (1 / Y.shape[1]) * \
            np.matmul(np.matmul(self.__W2.transpose(), A2 - Y) *
                      (A1 * (1 - A1)), X.transpose())
        db1 = (1 / Y.shape[1]) * \
            np.sum(np.matmul(self.__W2.transpose(), A2 - Y) *
                   (A1 * (1 - A1)), axis=1, keepdims=True)
        dw2 = (1 / Y.shape[1]) * np.matmul(A1, (A2 - Y).transpose())
        db2 = (1 / Y.shape[1]) * np.sum(A2 - Y, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dw2).transpose()
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neural network """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return self.evaluate(X, Y)
