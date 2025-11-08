#!/usr/bin/env python3
"""
Monte Carlo method for estimating the value function of a policy
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm
    Args:
        env: the environment to use
        V: the initial value function
        policy: the policy to follow
        episodes: the number of episodes to run
        max_steps: the maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount factor
    Returns:
        V, the updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        # Compute returns and update all visited states (every-visit MC)
        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] += alpha * (G - V[state])

    return V
