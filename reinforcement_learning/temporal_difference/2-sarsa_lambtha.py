#!/usr/bin/env python3
"""
SARSA(λ) algorithm implementation for on-policy Temporal Difference learning.

This module defines the function `sarsa_lambtha`, which implements
the SARSA(λ) algorithm with eligibility traces for updating the
action-value function (Q-table) in reinforcement learning.
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm.

    Args:
        env: the environment instance
        Q: numpy.ndarray of shape (s, a) containing the Q table
        lambtha: eligibility trace factor (λ)
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial epsilon for epsilon-greedy
        min_epsilon: minimum epsilon value
        epsilon_decay: decay rate for epsilon per episode

    Returns:
        Q: the updated Q table
    """
    n_states, n_actions = Q.shape

    for _ in range(episodes):
        # Reset environment and eligibility traces
        state, _ = env.reset()
        E = np.zeros((n_states, n_actions))

        # Choose initial action using epsilon-greedy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            # Perform the chosen action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose next action using epsilon-greedy
            if np.random.uniform() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            # Compute TD error
            td_error = (reward + gamma * Q[next_state, next_action]
                        - Q[state, action])

            # Update eligibility trace for (state, action)
            E[state, action] += 1

            # Update Q-values and decay traces
            Q += alpha * td_error * E
            E *= gamma * lambtha

            # If episode ends, break
            if terminated or truncated:
                break

            # Move to next step
            state = next_state
            action = next_action

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
