#!/usr/bin/env python3
"""
Train module for REINFORCE policy-gradient learning.

This module implements a training loop using:
- A softmax policy
- Monte Carlo REINFORCE updates
- Optional environment rendering every 1000 episodes

The agent is designed for environments such as CartPole-v1.
"""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Train a policy-gradient agent using the REINFORCE algorithm.

    Args:
        env (gymnasium.Env):
            The environment used for training.
        nb_episodes (int):
            Number of episodes to train the agent.
        alpha (float):
            Learning rate used to update the policy weights.
        gamma (float):
            Discount factor applied to future rewards.
        show_result (bool):
            If True, render the environment every 1000 episodes.

    Returns:
        list:
            A list containing the episode scores (sum of rewards per episode).
    """
    scores = []
    weight = np.random.rand(4, 2)  # For CartPole: 4 state dims, 2 actions

    for episode in range(nb_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_gradients = []

        done = False

        # Run one full episode
        while not done:
            action, grad = policy_gradient(state, weight)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_gradients.append(grad)
            episode_rewards.append(reward)

            state = next_state

        # Compute discounted returns
        discounted = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            discounted.insert(0, G)
        discounted = np.array(discounted)

        # Update weights (REINFORCE update rule)
        for grad, Gt in zip(episode_gradients, discounted):
            weight += alpha * grad * Gt

        score = sum(episode_rewards)
        scores.append(score)

        print(f"Episode: {episode} Score: {score}")

        # Render environment every 1000 episodes if requested
        if show_result and episode % 1000 == 0:
            env.render()

    return scores
