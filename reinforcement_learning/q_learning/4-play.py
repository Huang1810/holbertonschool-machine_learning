#!/usr/bin/env python3
"""
Module to let the trained agent play an episode on FrozenLake
using the existing 0-load_env.py without modifying it.
"""

import numpy as np

ACTION_NAMES = ["Left", "Down", "Right", "Up"]

def play(env, Q, max_steps=100):
    """
    Plays an episode using the Q-table and returns:
        - total_rewards: total reward earned in the episode
        - rendered_outputs: list of strings showing the board state
          at each step with the agent position quoted and actions shown
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []

    nrow, ncol = env.unwrapped.desc.shape

    for _ in range(max_steps):
        # Choose best action (exploit Q-table)
        action = np.argmax(Q[state])

        # Take the action
        next_state, reward, done, truncated, _ = env.step(action)
        total_rewards += reward

        # Build board string manually
        board_str = ""
        for r in range(nrow):
            row_str = ""
            for c in range(ncol):
                pos = r * ncol + c
                ch = env.unwrapped.desc[r, c].decode()
                if pos == state:
                    row_str += f'"{ch}"'
                else:
                    row_str += ch
            board_str += row_str
            if r != nrow - 1:
                board_str += "\n"

        # Append action just taken
        board_str += f"\n  ({ACTION_NAMES[action]})"
        rendered_outputs.append(board_str)

        state = next_state
        if done or truncated:
            break

    # Render final state (without appending an action)
    final_board = ""
    for r in range(nrow):
        row_str = ""
        for c in range(ncol):
            pos = r * ncol + c
            ch = env.unwrapped.desc[r, c].decode()
            if pos == state:
                row_str += f'"{ch}"'
            else:
                row_str += ch
        final_board += row_str
        if r != nrow - 1:
            final_board += "\n"
    final_board += "\n"  # Add final newline to match desired output
    rendered_outputs.append(final_board)

    return total_rewards, rendered_outputs
