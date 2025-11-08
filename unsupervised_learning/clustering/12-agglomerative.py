#!/usr/bin/env python3
"""
Performs agglomerative clustering with Ward linkage and displays a dendrogram.
"""

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np

def agglomerative(X, dist):
    """
    Performs agglomerative clustering on X with maximum cophenetic distance dist.

    Args:
        X (np.ndarray): Dataset of shape (n, d)
        dist (float): Maximum cophenetic distance for all clusters

    Returns:
        clss (np.ndarray): Cluster indices for each data point, shape (n,)
    """
    # Compute the linkage matrix using Ward's method
    Z = sch.linkage(X, method='ward')

    # Assign cluster labels based on the given distance threshold
    clss = sch.fcluster(Z, t=dist, criterion='distance')

    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    sch.dendrogram(Z, color_threshold=dist)
    plt.title('Agglomerative Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

    return np.array(clss)
