#!/usr/bin/env python3
"""
Forward Algorithm for Hidden Markov Model module.
This module contains a function that implements the forward algorithm to
calculate the likelihood of a sequence of observations given a hidden
Markov model.
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.
    """
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        # Initialize the forward path probabilities matrix
        F = np.zeros((N, T))

        # Initial step
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        # Recursion step
        for t in range(1, T):
            for j in range(N):
                F[j, t] = np.sum(
                    F[:, t-1] * Transition[:, j] * Emission[j, Observation[t]]
                )

        # Termination step
        P = np.sum(F[:, T-1])

        return P, F

    except Exception:
        return None, None
