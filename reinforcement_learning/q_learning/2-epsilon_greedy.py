#!/usr/bin/env python3
"""
Module that implements the epsilon-greedy policy
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q: numpy.ndarray containing the Q-table
        state: the current state
        epsilon: the epsilon value for exploration vs exploitation

    Returns:
        The next action index
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        # Explore: choose a random action
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit: choose the action with the highest Q-value
        action = np.argmax(Q[state])

    return action
