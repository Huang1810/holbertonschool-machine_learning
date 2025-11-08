#!/usr/bin/env python3
import numpy as np


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm.

    Args:
        env: the environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes a state and returns the next action
        lambtha: the eligibility trace factor (λ)
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    n_states = V.shape[0]

    for _ in range(episodes):
        # Reset environment and eligibility traces
        state, _ = env.reset()
        E = np.zeros(n_states)  # eligibility trace vector

        for _ in range(max_steps):
            # Select action using policy
            action = policy(state)

            # Take the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Compute TD error
            delta = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace
            E[state] += 1

            # Update all states' values
            V += alpha * delta * E

            # Decay eligibility traces
            E *= gamma * lambtha

            # If episode ends, break
            if terminated or truncated:
                break

            # Move to next state
            state = next_state

    return V
