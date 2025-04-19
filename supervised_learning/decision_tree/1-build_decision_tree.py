#!/usr/bin/env python3
"""
Defines the structure for building a simple decision tree,
including Node, Leaf, and Decision_Tree classes.
"""
import numpy as np


class Node:
    """
    A decision node that splits data based on a feature and threshold.

    Attributes:
        feature (int): Feature index used for splitting.
        threshold (float): Value to split the feature on.
        left_child (Node): Left subtree.
        right_child (Node): Right subtree.
        is_root (bool): True if node is the tree's root.
        depth (int): Node depth in the tree.
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

    def max_depth_below(self):
        """
        Returns the maximum depth of all nodes below this one.
        """
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts all nodes (or only leaves) below this node.

        Args:
            only_leaves (bool): If True, counts only leaf nodes.

        Returns:
            int: Number of nodes or leaves below.
        """
        if only_leaves:
            if self.is_leaf:
                return 1
            count = 0
        else:
            count = 1  # Include this node

        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count


class Leaf(Node):
    """
    A terminal node that stores the prediction result.

    Attributes:
        value (any): Predicted value.
        depth (int): Depth of this leaf.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of this leaf (no children).
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 since a leaf is a terminal node.

        Args:
            only_leaves (bool): Ignored for leaves.

        Returns:
            int: Always 1.
        """
        return 1


class Decision_Tree:
    """
    A basic decision tree model.

    Attributes:
        max_depth (int): Max depth allowed for the tree.
        min_pop (int): Min samples required to split a node.
        seed (int): RNG seed for reproducibility.
        split_criterion (str): Strategy for splitting nodes.
        root (Node): The root node of the tree.
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

    def depth(self):
        """
        Returns the tree’s maximum depth.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the total number of nodes (or just leaves) in the tree.

        Args:
            only_leaves (bool): If True, counts only leaves.

        Returns:
            int: Number of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
