#!/usr/bin/env python3
"""
Function epsilon_greedy
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Returns: the next action index
    """
    exploration_rate_threshold = np.random.uniform(0, 1)
    if exploration_rate_threshold > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, 3, None)
    return action
