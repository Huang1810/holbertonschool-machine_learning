#!/usr/bin/env python3
"""
Module to let the trained agent play an episode on FrozenLake
with proper board rendering and action display.
"""

import numpy as np

ACTION_NAMES = ["Left", "Down", "Right", "Up"]

def play(env, Q, max_steps=100):
    """
    Plays an episode using the Q-table, returns total rewards
    and a list of rendered outputs with quoted current positions.
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []

    nrow, ncol = env.unwrapped.desc.shape

    for step in range(max_steps):
        # Build board string
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

        # Append previous action except for first step
        if step > 0:
            board_str += f"\n  ({ACTION_NAMES[action]})"

        rendered_outputs.append(board_str)

        # Choose best action (exploit)
        action = np.argmax(Q[state])

        # Take action
        next_state, reward, done, truncated, _ = env.step(action)
        total_rewards += reward
        state = next_state

        if done or truncated:
            break

    # Render final state
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
    rendered_outputs.append(final_board)

    return total_rewards, rendered_outputs
