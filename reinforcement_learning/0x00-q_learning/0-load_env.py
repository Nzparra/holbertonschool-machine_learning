#!/usr/bin/env python3
"""
the pre-made FrozenLakeEnv
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
        Returns: the environment
    """
    env = gym.make('FrozenLake-v0',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env
