#!/usr/bin/env python3
"""
Module to let the trained agent play an episode on FrozenLake.
"""

import numpy as np

ACTION_NAMES = ["Left", "Down", "Right", "Up"]

def play(env, Q, max_steps=100):
    """
    Plays an episode using the trained Q-table and returns
    total rewards and rendered board states with actions.
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []

    for step in range(max_steps):
        # Render board as string
        board = env.render()

        # Append the last action taken except on the first step
        if step > 0:
            board += f"\n  ({ACTION_NAMES[action]})"
        rendered_outputs.append(board)

        # Choose best action (exploit)
        action = np.argmax(Q[state])

        # Take the action
        next_state, reward, done, truncated, _ = env.step(action)
        total_rewards += reward
        state = next_state

        if done or truncated:
            break

    # Render final state
    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
