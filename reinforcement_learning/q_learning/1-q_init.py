#!/usr/bin/env python3
"""
Module that initializes the Q-table for a FrozenLake environment
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table for a given FrozenLake environment.

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        A numpy.ndarray of zeros with shape (number of states, number of actions)
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    return Q
