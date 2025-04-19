#!/usr/bin/env python3
"""
Module for the Isolation_Random_Tree class. This class is designed for anomaly detection
by isolating data points within a tree structure. Anomalies are typically isolated closer
to the root of the tree, and the method does not require prior knowledge of the distribution
of normal data points.
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    An isolation tree used for anomaly detection. Unlike traditional decision trees,
    this tree isolates data points by creating partitions that result in anomalies being
    isolated at shallower depths.

    Attributes:
        rng (np.random.Generator): Random number generator for splitting data.
        root (Node): The root node of the isolation tree.
        explanatory (np.ndarray): The feature data used for fitting.
        max_depth (int): Maximum allowable depth for the tree.
        predict (callable): Function used for making predictions.
        min_pop (int): Minimum number of samples required for splitting during training.
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initializes the Isolation_Random_Tree with the provided maximum depth and random seed.

        Args:
            max_depth (int): Maximum depth the tree can reach during training.
            seed (int): Seed for random number generation for reproducibility.
            root (Node, optional): The starting node for the tree (if continuing from an existing node).
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Returns a string representation of the isolation tree.

        Returns:
            str: A string description of the tree's structure.
        """
        return self.root.__str__()

    def depth(self):
        """
        Calculates the maximum depth of the isolation tree.

        Returns:
            int: The maximum depth from the root to any leaf node.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the total number of nodes in the isolation tree, with an option to count only leaf nodes.

        Args:
            only_leaves (bool): If True, only counts leaf nodes.

        Returns:
            int: The total number of nodes or leaf nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """
        Retrieves all the leaf nodes in the isolation tree.

        Returns:
            list: A list of all leaf nodes.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Begins a recursive process to update the bounds of the tree, starting from the root node.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Updates the prediction function for the isolation tree.
        The function is modified based on the current state of the tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([self.root.pred(x) for x in A])

    def np_extrema(self, arr):
        """
        Retrieves the minimum and maximum values from the provided array.

        Args:
            arr (np.ndarray): The input array.

        Returns:
            tuple: The minimum and maximum values of the array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly selects a feature and a threshold for splitting a node.

        Args:
            node (Node): The node that will be split.

        Returns:
            tuple: A pair consisting of the chosen feature index and the threshold value.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Creates and returns a leaf child node for a given parent node.

        Args:
            node (Node): The parent node.
            sub_population (np.ndarray): The indices of the samples that belong to this leaf.

        Returns:
            Leaf: A newly created leaf node.
        """
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates a new child node for further splitting during tree construction.

        Args:
            node (Node): The parent node.
            sub_population (array): The subset of data points that belong to the new node.

        Returns:
            Node: A newly created child node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively trains the isolation tree, starting from the specified node.

        Args:
            node (Node): The node to train from.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        above_threshold = self.explanatory[:, node.feature] > node.threshold
        left_population = node.sub_population & above_threshold
        right_population = node.sub_population & ~above_threshold

        # Determine if left or right child should be a leaf
        is_left_leaf = np.any([node.depth >= self.max_depth - 1,
                               np.sum(left_population) <= self.min_pop])

        # Create left child
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Determine if right child should be a leaf
        is_right_leaf = np.any([node.depth >= self.max_depth - 1,
                                np.sum(right_population) <= self.min_pop])

        # Create right child
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Fits the isolation tree to the provided data.

        Args:
            explanatory (np.ndarray): The data to be used for training the tree.
            verbose (int): A flag to control verbosity. If 1, detailed information is printed during training.
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training completed.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }""")
