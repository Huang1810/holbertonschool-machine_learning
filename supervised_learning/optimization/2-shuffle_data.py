#!/usr/bin/env python3
"""
Shuffles the data points in two matrices the same way.
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    """
    perm = np.random.permutation(X.shape[0])

    X_shuffled = X[perm]
    Y_shuffled = Y[perm]
    return X_shuffled, Y_shuffled
