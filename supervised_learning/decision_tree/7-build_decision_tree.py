#!/usr/bin/env python3
"""
This module defines the classes for building a basic decision tree, 
including Node, Leaf, and Decision_Tree. The decision tree can be 
trained and used for prediction using recursive splitting strategies.
"""
import numpy as np


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting.
        threshold (float): Threshold value for the split.
        left_child (Node): Left child node.
        right_child (Node): Right child node.
        is_root (bool): Indicates if the node is the root.
        depth (int): Depth of the node in the tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initializes a new node for the decision tree.
        
        Args:
            feature (int): Feature index to split on.
            threshold (float): Threshold value to split the data.
            left_child (Node): Left child node.
            right_child (Node): Right child node.
            is_root (bool): Flag indicating if this node is the root.
            depth (int): Depth of this node in the tree.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def __str__(self):
        """
        Returns a string representation of the node and its children.

        Returns:
            str: A string representation of the subtree rooted at this node.
        """
        p = "root" if self.is_root else "-> node"
        result = f"{p} [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            result += self.left_child_add_prefix(self.left_child.__str__().strip())
        if self.right_child:
            result += self.right_child_add_prefix(self.right_child.__str__().strip())
        return result

    def left_child_add_prefix(self, text):
        """
        Adds a prefix to the left child's subtree representation.

        Args:
            text (str): The subtree string to which the prefix will be added.

        Returns:
            str: Modified subtree string with added prefixes for left child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds a prefix to the right child's subtree representation.

        Args:
            text (str): The subtree string to which the prefix will be added.

        Returns:
            str: Modified subtree string with added prefixes for right child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "       " + x + "\n"
        return new_text

    def max_depth_below(self):
        """
        Computes the maximum depth of the tree starting from this node.

        Returns:
            int: Maximum depth below this node.
        """
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the nodes below this node, optionally only counting leaves.

        Args:
            only_leaves (bool): If True, only counts leaf nodes.

        Returns:
            int: The count of nodes or leaves under this node.
        """
        count = 1 if not only_leaves or self.is_leaf else 0

        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def get_leaves_below(self):
        """
        Returns all leaf nodes below this node.

        Returns:
            list: A list of all leaf nodes in the subtree rooted at this node.
        """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            if self.left_child:
                leaves.extend(self.left_child.get_leaves_below())
            if self.right_child:
                leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively updates bounds for the node and its children.
        Initializes at root with infinite bounds and adjusts for children 
        based on feature values.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                elif child == self.right_child:
                    child.upper[self.feature] = self.threshold
        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Updates the indicator function to determine whether an instance's 
        features meet the node's criteria based on its bounds.
        """
        def is_large_enough(x):
            return np.array([np.greater_equal(x[:, key], self.lower[key])
                             for key in self.lower.keys()]).all(axis=0)

        def is_small_enough(x):
            return np.array([np.less_equal(x[:, key], self.upper[key])
                             for key in self.upper.keys()]).all(axis=0)

        self.indicator = lambda x: np.logical_and(is_large_enough(x),
                                                  is_small_enough(x))

    def update_predict(self):
        """
        Updates the prediction function for the tree by setting the indicator 
        for each leaf and preparing the tree to make predictions.
        """
        self.update_bounds_below()
        leaves = self.get_leaves_below()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([
            leaf.value for x in A for leaf in leaves if leaf.indicator(x)
        ])

    def pred(self, x):
        """
        Recursively predicts the output for a sample by traversing down 
        the tree based on input features.

        Args:
            x (array): Input features for a single sample.

        Returns:
            any: The predicted value.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value (any): The value predicted by this leaf.
        depth (int): Depth of the leaf in the tree.
    """
    def __init__(self, value, depth=None):
        """
        Initializes a leaf node with a predicted value.

        Args:
            value (any): The predicted value of the leaf.
            depth (int): Depth of this leaf in the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Returns a string representation of the leaf.

        Returns:
            str: A string representing the leaf.
        """
        return f"-> leaf [value={self.value}]"

    def max_depth_below(self):
        """
        Returns the depth of the leaf, as leaves are terminal nodes.

        Returns:
            int: Depth of the leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1, as this node is a leaf and is always counted.

        Args:
            only_leaves (bool): Ignored as a leaf is always counted.

        Returns:
            int: Always 1 for this leaf node.
        """
        return 1

    def get_leaves_below(self):
        """
        Returns a list containing only this leaf node.

        Returns:
            list: A list containing the leaf node.
        """
        return [self]

    def update_bounds_below(self):
        """
        No bounds update for leaf nodes.
        """
        pass

    def pred(self, x):
        """
        Returns the value stored in the leaf as the prediction.

        Args:
            x (array): Input features (ignored for leaf nodes).

        Returns:
            any: The predicted value of the leaf node.
        """
        return self.value


class Decision_Tree:
    """
    Represents a decision tree for classification or regression.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_pop (int): Minimum population required to split a node.
        seed (int): Random seed for reproducibility.
        split_criterion (str): Splitting criterion for node splitting.
        root (Node): The root node of the tree.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0, 
                 split_criterion="random", root=None):
        """
        Initializes the decision tree with the given parameters.

        Args:
            max_depth (int): Maximum depth for the tree.
            min_pop (int): Minimum population for splitting nodes.
            seed (int): Random seed for reproducibility.
            split_criterion (str): Criterion for splitting nodes.
            root (Node): The root node of the tree.
        """
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.split_criterion = split_criterion
        self.root = root if root else Node(is_root=True)
