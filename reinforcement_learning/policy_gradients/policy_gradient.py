#!/usr/bin/env python3
import numpy as np


def policy(state, weight):
    """
    Computes the softmax policy.
    state: numpy array (1, n) or (n,)
    weight: numpy array (n, m)
    Return: softmax probability of each action
    """
    # Ensure row vector
    state = np.array(state).reshape(1, -1)

    # Compute logits
    z = np.dot(state, weight)

    # Softmax (numerically stable)
    exp = np.exp(z - np.max(z))
    softmax = exp / np.sum(exp, axis=1, keepdims=True)
    return softmax


def policy_gradient(state, weight):
    """
    Computes Monte-Carlo policy gradient for a single state.
    state: vector (n,)
    weight: matrix (n, m)
    Return: (action, gradient)
    """
    # Get policy probabilities
    probs = policy(state, weight).flatten()

    # Sample an action using the probabilities
    action = np.random.choice(len(probs), p=probs)

    # One-hot encode action
    action_one_hot = np.zeros_like(probs)
    action_one_hot[action] = 1

    # Reshape state into column vector
    s = np.array(state).reshape(-1, 1)

    # Gradient of log π(a|s)
    grad = np.dot(s, (action_one_hot - probs).reshape(1, -1))

    return action, grad
