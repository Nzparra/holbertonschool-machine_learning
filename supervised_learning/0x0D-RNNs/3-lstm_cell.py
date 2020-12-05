#!/usr/bin/env python3
""" Long Short Term Memory """

import numpy as np


class LSTMCell():
    """ LSTM cell """

    def __init__(self, i, h, o):
        """
        * The weights will be used on the right side for matrix
          multiplication
        * The biases should be initialized as zeros
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Returns: h_next, c_next, y
        """
        xh = np.concatenate((h_prev, x_t), axis=1)
        ft = np.dot(xh, self.Wf) + self.bf
        ft = 1 / (1 + np.exp(-ft))
        ut = np.dot(xh, self.Wu) + self.bu
        ut = 1 / (1 + np.exp(-ut))
        c_hat = np.tanh(np.dot(xh, self.Wc) + self.bc)
        c_next = ft * c_prev + ut * c_hat
        ot = np.dot(xh, self.Wo) + self.bo
        ot = 1 / (1 + np.exp(-ot))
        h_next = ot * np.tanh(c_next)
        y_pred = np.dot(h_next, self.Wy) + self.by
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        return (h_next, c_next, y_pred)
