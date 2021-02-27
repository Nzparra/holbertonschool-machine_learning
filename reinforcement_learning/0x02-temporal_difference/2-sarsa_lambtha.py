#!/usr/bin/env python3
"""
* https://towardsdatascience.com/eligibility-traces
  in-reinforcement-learning-a6b458c019d6
Sarsa with eligibility trace
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Returns: the next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Returns: Q, the updated Q table
    """
    for i in range(episodes):
        state = env.reset()
        e = np.zeros((Q.shape))
        action = epsilon_greedy(Q, state, epsilon=epsilon)
        for j in range(max_steps):
            new_state, reward, done, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon=epsilon)
            f = reward + gamma * Q[new_state, new_action] - Q[state, action]
            e[state, action] += 1

            Q = Q + alpha * f * e
            e = e * lambtha * gamma
            state = new_state
            action = new_action

        if done:
            break
        part = (1 - min_epsilon) * np.exp(-epsilon_decay * i)
        epsilon = min_epsilon + part
    return Q
