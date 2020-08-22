#!/usr/bin/env python3
""" updates a variable using the RMSProp optimization algorithm """

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
        loss is the loss of the network
        alpha is the learning rate
        beta1 is the momentum weight
        Returns: the momentum optimization operation
    """

    return tf.train.MomentumOptimizer(learning_rate=alpha,
                                      momentum=beta1).minimize(loss=loss)
