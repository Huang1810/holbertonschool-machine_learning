#!/usr/bin/env python3
"""
Policy gradient module.

This module implements:
- policy: a softmax policy function
- policy_gradient: Monte Carlo REINFORCE policy gradient

Both functions operate on NumPy arrays and are used to train
a policy-gradient agent for environments like CartPole-v1.
"""

import numpy as np


def policy(state, weight):
    """
    Compute the softmax policy for a given state and weight matrix.

    Args:
        state (numpy.ndarray): Observation/state of shape (4,) or (1, 4).
        weight (numpy.ndarray): Weight matrix of shape (4, 2).

    Returns:
        numpy.ndarray: Softmax probability distribution over actions
                       with shape (1, 2).
    """
    state = np.array(state).reshape(1, -1)

    z = np.dot(state, weight)
    exp = np.exp(z - np.max(z))
    softmax = exp / np.sum(exp, axis=1, keepdims=True)

    return softmax


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo REINFORCE policy gradient for a single step.

    Args:
        state (numpy.ndarray): Environment observation/state (size 4).
        weight (numpy.ndarray): Weight matrix of shape (4, 2).

    Returns:
        tuple:
            - int: the sampled action (0 or 1)
            - numpy.ndarray: gradient of the log-policy w.r.t. weights
                             with shape (4, 2)
    """
    probs = policy(state, weight).flatten()

    action = np.random.choice(len(probs), p=probs)

    action_one_hot = np.zeros_like(probs)
    action_one_hot[action] = 1

    s = np.array(state).reshape(-1, 1)

    grad = np.dot(s, (action_one_hot - probs).reshape(1, -1))

    return action, grad
