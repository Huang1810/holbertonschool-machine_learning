#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value of each state.

    Args:
        env: the environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    for _ in range(episodes):
        # Generate one episode following the policy
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        # Compute returns (G) for each state in the episode
        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            # Incremental mean update rule
            V[state] = V[state] + alpha * (G - V[state])

    return V
