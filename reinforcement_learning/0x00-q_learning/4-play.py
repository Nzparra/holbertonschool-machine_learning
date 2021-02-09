#!/usr/bin/env python3
"""
function play
"""
import numpy as np
load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
train = __import__('3-q_learning').train


def play(env, Q, max_steps=100):
    """
    Returns total rewards for the episode
    """
    state = env.reset()
    done = False
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        env.render()
        if done is True:
            if reward == 1:
                print(reward)
                break
        state = new_state
    env.close()
