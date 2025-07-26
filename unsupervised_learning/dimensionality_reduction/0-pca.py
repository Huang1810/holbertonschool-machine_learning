#!/usr/bin/env python3
"""PCA function"""

import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset while retaining a given fraction of variance.

    Args:
        X (np.ndarray): shape (n, d), centered dataset (zero mean)
        var (float): desired fraction of variance to retain (0 < var <= 1)

    Returns:
        np.ndarray: shape (d, nd), weight matrix with nd principal components
    """
    # Perform SVD
    u, s, vh = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance ratio
    explained_variance = (s ** 2) / np.sum(s ** 2)
    cumulative_variance = np.cumsum(explained_variance)

    # Number of components to retain desired variance
    r = np.searchsorted(cumulative_variance, var) + 1

    # Return projection matrix (d, r)
    return vh[:r].T
