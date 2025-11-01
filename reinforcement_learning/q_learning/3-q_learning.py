#!/usr/bin/env python3
"""
Module that trains an agent using Q-learning on FrozenLake
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning on a FrozenLake environment.

    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial epsilon for exploration
        min_epsilon: minimum epsilon value
        epsilon_decay: rate of epsilon decay

    Returns:
        Q: updated Q-table
        total_rewards: list containing the rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps):
            # Choose action using epsilon-greedy
            action = epsilon_greedy(Q, state, epsilon)

            # Take action in the environment
            new_state, reward, done, truncated, info = env.step(action)

            # If the agent falls in a hole, penalize
            if reward == 0 and done:
                reward = -1

            # Update Q-value using the Q-learning update rule
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            state = new_state
            total_reward += reward

            if done or truncated:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))
        total_rewards.append(total_reward)

    return Q, total_rewards
