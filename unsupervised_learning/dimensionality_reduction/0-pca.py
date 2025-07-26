#!/usr/bin/env python3
"""PCA function"""

import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset to reduce its dimensionality
    while retaining a specified fraction of variance.

    Args:
        X (np.ndarray): shape (n, d), centered dataset
        var (float): fraction of variance to preserve

    Returns:
        W (np.ndarray): shape (d, nd), projection matrix
    """
    # Perform SVD
    u, s, vh = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance from singular values
    explained_variance = (s ** 2) / np.sum(s ** 2)
    cumulative_variance = np.cumsum(explained_variance)

    # Get number of components needed to reach the desired variance
    r = np.argmax(cumulative_variance >= var) + 1

    # Construct the weight matrix
    W = vh[:r].T  # Shape: (d, r)
    return W
