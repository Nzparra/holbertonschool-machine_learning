#!/usr/bin/env python3
""" learning rate using inverse time decay in numpy """


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        alpha is the original learning rate
        decay_rate is the weight used to determine the rate
        global_step is the number of passes of gradient descent
        decay_step is the number of passes of gradient descent
        the learning rate decay should occur in a stepwise fashion
        Returns: the updated value for alpha
    """
    return (alpha / (1 + (decay_rate * (global_step // decay_step))))
