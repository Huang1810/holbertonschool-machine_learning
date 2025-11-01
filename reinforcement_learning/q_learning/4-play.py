#!/usr/bin/env python3
"""
Module that allows a trained Q-learning agent to play FrozenLake
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Lets the trained agent play an episode using the Q-table.

    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: total reward earned in the episode
        rendered_outputs: list of strings showing the board state at each step
    """
    state, _ = env.reset()
    done = False
    total_rewards = 0
    rendered_outputs = []

    for _ in range(max_steps):
        # Always exploit: choose best action based on Q-table
        action = np.argmax(Q[state])

        # Render the current environment and store the string output
        rendered_outputs.append(env.render())

        # Take the chosen action
        new_state, reward, done, truncated, info = env.step(action)

        state = new_state
        total_rewards += reward

        if done or truncated:
            break

    # Ensure the final state is rendered
    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
