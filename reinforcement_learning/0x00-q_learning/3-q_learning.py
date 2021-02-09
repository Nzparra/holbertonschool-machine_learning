#!/usr/bin/env python3
"""
Q-learning:
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Returns: Q, total_rewards
    """
    total_rewards = []
    for epi in range(episodes):
        state = env.reset()
        done = False
        rewards = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            Q[state, action] = (Q[state, action] * (1 - alpha) +
                                (reward + gamma*np.max(Q[new_state, :]))*alpha)
            state = new_state
            if done is True:
                if reward != 1:
                    rewards = -1
                rewards += reward
                break
            else:
                rewards += reward
        total_rewards.append(rewards)
        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-epsilon_decay*epi)
    return Q, total_rewards
