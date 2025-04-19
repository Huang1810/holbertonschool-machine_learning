#!/usr/bin/env python3
"""
This module defines classes for constructing a decision tree, including
Node, Leaf, and Decision_Tree, for use in decision-making processes.
"""
import numpy as np


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting the data.
        threshold (float): The threshold value for the split.
        left_child (Node): The left child node after the split.
        right_child (Node): The right child node after the split.
        is_root (bool): Flag indicating if this node is the root node.
        depth (int): The depth of this node in the tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
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
            str: A string that describes the subtree rooted at this node.
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
            text (str): The subtree string to be modified.

        Returns:
            str: The modified subtree string with added prefixes for left child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds a prefix to the right child's subtree representation.

        Args:
            text (str): The subtree string to be modified.

        Returns:
            str: The modified subtree string with added prefixes for right child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def max_depth_below(self):
        """
        Calculates the maximum depth of the tree starting from this node.

        Returns:
            int: The maximum depth of the subtree rooted at this node.
        """
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes (or leaves) in the subtree rooted at this node.

        Args:
            only_leaves (bool): If True, counts only leaf nodes.

        Returns:
            int: The total number of nodes (or leaves) below this node.
        """
        if only_leaves:
            if self.is_leaf:
                return 1
            count = 0
        else:
            count = 1  # Count this node

        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def get_leaves_below(self):
        """
        Retrieves all the leaf nodes in the subtree rooted at this node.

        Returns:
            list: A list containing all the leaf nodes in the subtree.
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
        Recursively updates the bounds (upper and lower) for the current node
        and its descendants. Starts with infinite bounds at the root and
        adjusts for each child node.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                # Copy the current node's bounds to each child.
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
        Updates the indicator function for this node based on its bounds.
        This function determines whether a given set of features meets the
        node's criteria for splitting.
        """
        def is_large_enough(x):
            return np.array([np.greater_equal(x[:, key], self.lower[key])
                             for key in self.lower.keys()]).all(axis=0)

        def is_small_enough(x):
            return np.array([np.less_equal(x[:, key], self.upper[key])
                             for key in self.upper.keys()]).all(axis=0)

        self.indicator = lambda x: np.logical_and(is_large_enough(x),
                                                  is_small_enough(x))


class Leaf(Node):
    """
    Represents a leaf node in the decision tree.

    Attributes:
        value (any): The value predicted by this leaf.
        depth (int): Depth of the leaf node in the tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Returns a string representation of this leaf node.

        Returns:
            str: A string that describes the leaf node.
        """
        return f"-> leaf [value={self.value}] "

    def max_depth_below(self):
        """
        Returns the depth of this leaf node.

        Returns:
            int: The depth of this leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the leaf nodes under this node.

        Args:
            only_leaves (bool): In the case of a leaf, this flag is ignored
            as this node is always counted.

        Returns:
            int: Always returns 1 since a leaf node counts as one.
        """
        return 1

    def get_leaves_below(self):
        """
        Returns a list containing only this leaf node.

        Returns:
            list: A list containing only this leaf node.
        """
        return [self]

    def update_bounds_below(self):
        """
        Placeholder method for leaves, as they do not have child nodes to update.
        """
        pass


class Decision_Tree:
    """
    Represents a decision tree classifier.

    Attributes:
        max_depth (int): Maximum depth allowed for the tree.
        min_pop (int): Minimum number of data points required to split a node.
        seed (int): Random seed for reproducibility.
        split_criterion (str): Criterion used for deciding the best split.
        root (Node): The root node of the decision tree.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        """
        Returns a string representation of the decision tree.

        Returns:
            str: A string that represents the entire decision tree.
        """
        return self.root.__str__()

    def depth(self):
        """
        Computes the maximum depth of the tree.

        Returns:
            int: The maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the nodes in the entire tree.

        Args:
            only_leaves (bool): If True, counts only the leaf nodes.

        Returns:
            int: The total number of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """
        Returns a list of all leaf nodes in the tree.

        Returns:
            list: A list of all leaf nodes in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Starts the recursive update of bounds from the root node.
        """
        self.root.update_bounds_below()

    def update_indicator(self):
        """
        Updates the indicator functions for all nodes in the tree.
        """
        self.root.update_indicator()
