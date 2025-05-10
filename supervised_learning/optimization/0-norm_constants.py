#!/usr/bin/env python3
"""
This module provides a function to compute normalization constants
(mean and standard deviation) for a given NumPy matrix.
"""

import numpy as np


def normalization_constants(X):
    """
    Calculate the normalization constants (mean and standard deviation)
    of a matrix
    """
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    return mean, std_dev
