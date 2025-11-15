#!/usr/bin/env python3
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Train a REINFORCE agent on CartPole.
    
    env: Gym environment
    nb_episodes: number of training episodes
    alpha: learning rate
    gamma: discount factor
    show_result: if True, render every 1000 episodes
    Return: list of episode scores
    """
    scores = []

    # Initialize random weights: (state_size=4, action_size=2)
    weight = np.random.rand(4, 2)

    for episode in range(nb_episodes):
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []

        done = False

        # Collect one full episode
        while not done:
            action, grad = policy_gradient(state, weight)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_actions.append(grad)
            episode_rewards.append(reward)

            state = next_state

        # Compute discounted rewards
        discounted = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            discounted.insert(0, G)

        discounted = np.array(discounted)

        # Update weights using REINFORCE rule
        for g, Gt in zip(episode_actions, discounted):
            weight += alpha * g * Gt

        score = sum(episode_rewards)
        scores.append(score)

        # Print required output
        print(f"Episode: {episode} Score: {score}")

        # Render if show_result=True and episode % 1000 == 0
        if show_result and episode % 1000 == 0:
            env.render()

    return scores
