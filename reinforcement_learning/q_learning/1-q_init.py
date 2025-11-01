#!/usr/bin/env python3
"""
Module that initializes the Q-table
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table.

    Args:
        env: FrozenLakeEnv instance

    Returns:
        Q-table as a NumPy array of zeros
    """
    return np.zeros((env.observation_space.n,
                     env.action_space.n))
