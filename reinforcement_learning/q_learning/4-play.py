#!/usr/bin/env python3
"""
Module that allows a trained Q-learning agent to play FrozenLake
"""

import numpy as np

ACTION_NAMES = ["Left", "Down", "Right", "Up"]

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
        # Always exploit
        action = np.argmax(Q[state])

        # Render board
        board = env.render()
        # Add action taken if this is not the first step
        if rendered_outputs:
            board += f"\n  ({ACTION_NAMES[action]})"
        rendered_outputs.append(board)

        # Take step
        new_state, reward, done, truncated, info = env.step(action)
        total_rewards += reward
        state = new_state

        if done or truncated:
            break

    # Render final state
    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
