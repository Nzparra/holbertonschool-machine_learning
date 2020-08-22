#!/usr/bin/env python3
""" learning rate using inverse time decay in numpy """

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        alpha is the original learning rate
        decay_rate is the weight used to determine the rate
        global_step is the number of passes of gradient descent
        decay_step is the number of passes of gradient descent
        the learning rate decay should occur in a stepwise fashion
        Returns: the updated value for alpha
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)
