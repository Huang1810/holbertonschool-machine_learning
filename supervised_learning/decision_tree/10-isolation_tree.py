#!/usr/bin/env python3
"""
Module implementing Isolation Random Tree for outlier detection using
Isolation Trees.
Designed for high-dimensional datasets, it isolates observations based on
random feature selection and splitting.
"""
import numpy as np
import random


class Isolation_Random_Tree:
    """
    A class that implements an Isolation Tree for anomaly detection.
    An Isolation Tree isolates observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        seed (int): Random seed for reproducible results.
        tree (dict): The structure of the tree.
    """

    def __init__(self, max_depth=10, seed=0):
        """
        Initializes the Isolation Random Tree with the specified parameters.

        Args:
            max_depth (int): Maximum depth of the tree.
            seed (int): Seed for the random number generator.
        """
        self.max_depth = max_depth
        self.seed = seed
        self.tree = {}

    def fit(self, data):
        """
        Fits the Isolation Tree to the provided dataset.

        Args:
            data (numpy.ndarray): The dataset to fit the tree.
        
        Returns:
            None
        """
        random.seed(self.seed)
        self.tree = self._build_tree(data, depth=0)

    def _build_tree(self, data, depth):
        """
        Recursively builds the isolation tree.

        Args:
            data (numpy.ndarray): The dataset to build the tree on.
            depth (int): Current depth of the tree.
        
        Returns:
            dict: The tree structure.
        """
        n_samples, n_features = data.shape
        if depth >= self.max_depth or n_samples <= 1:
            return {'depth': depth, 'size': n_samples}

        feature = random.randint(0, n_features - 1)
        min_val, max_val = data[:, feature].min(), data[:, feature].max()
        split_value = random.uniform(min_val, max_val)

        left_data = data[data[:, feature] <= split_value]
        right_data = data[data[:, feature] > split_value]

        return {
            'feature': feature,
            'split_value': split_value,
            'left': self._build_tree(left_data, depth + 1),
            'right': self._build_tree(right_data, depth + 1),
        }

    def predict(self, data):
        """
        Predicts the anomaly score for each data point.

        Args:
            data (numpy.ndarray): The dataset to predict anomalies on.

        Returns:
            numpy.ndarray: The anomaly scores for each sample.
        """
        return np.array([self._predict_sample(sample) for sample in data])

    def _predict_sample(self, sample):
        """
        Predicts the anomaly score for a single sample.

        Args:
            sample (numpy.ndarray): The data sample to predict.

        Returns:
            int: The depth at which the sample is isolated.
        """
        node = self.tree
        depth = 0

        while isinstance(node, dict) and 'feature' in node:
            feature = node['feature']
            split_value = node['split_value']
            if sample[feature] <= split_value:
                node = node['left']
            else:
                node = node['right']
            depth += 1

        return depth

    def depth(self):
        """
        Returns the maximum depth of the tree.

        Returns:
            int: The maximum depth.
        """
        return self._get_depth(self.tree)

    def _get_depth(self, node):
        """
        Recursively calculates the depth of the tree.

        Args:
            node (dict): The current node of the tree.

        Returns:
            int: The depth of the tree.
        """
        if isinstance(node, dict) and 'left' in node and 'right' in node:
            left_depth = self._get_depth(node['left'])
            right_depth = self._get_depth(node['right'])
            return max(left_depth, right_depth) + 1
        return node['depth']

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the tree.

        Args:
            only_leaves (bool): If True, counts only the leaves of the tree.

        Returns:
            int: The number of nodes (or leaves) in the tree.
        """
        return self._count_nodes(self.tree, only_leaves)

    def _count_nodes(self, node, only_leaves=False):
        """
        Recursively counts the number of nodes in the tree.

        Args:
            node (dict): The current node of the tree.
            only_leaves (bool): If True, counts only the leaves.

        Returns:
            int: The number of nodes (or leaves) in the tree.
        """
        if isinstance(node, dict):
            if only_leaves and 'left' not in node and 'right' not in node:
                return 1
            return 1 + self._count_nodes(node.get('left', {}), only_leaves) + \
                self._count_nodes(node.get('right', {}), only_leaves)
        return 0
