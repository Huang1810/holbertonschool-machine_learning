#!/usr/bin/env python3
"""PCA function"""

import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset to reduce its dimensionality while retaining a given variance.
    
    Args:
        X: numpy.ndarray of shape (n, d)
           - n: number of data points
           - d: number of dimensions/features
           Assumes X is already centered (zero mean).
        var: float
            - Fraction of variance to retain (between 0 and 1)

    Returns:
        W: numpy.ndarray of shape (d, nd)
           - Weight matrix for projecting the data
           - nd is the number of principal components selected
    """
    # Perform Singular Value Decomposition
    u, s, vh = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance
    explained_variance = (s ** 2) / np.sum(s ** 2)
    cumulative_variance = np.cumsum(explained_variance)

    # Find the number of components to retain enough variance
    r = np.searchsorted(cumulative_variance, var) + 1

    # Return the first r principal components
    W = vh[:r].T
    return W
