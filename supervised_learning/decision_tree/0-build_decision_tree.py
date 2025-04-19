#!/usr/bin/env python3
"""
Defines the basic structure for building a decision tree,
including Node, Leaf, and Decision_Tree classes.
"""
import numpy as np


class Node:
    """
    A decision node in the tree that splits based on a feature and threshold.

    Attributes:
        feature (int): Index of feature used for the split.
        threshold (float): Split threshold value.
        left_child (Node): Left child node.
        right_child (Node): Right child node.
        is_root (bool): True if this node is the root.
        depth (int): Depth of this node in the tree.
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
        Returns the maximum depth of all branches below this node.
        """
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth


class Leaf(Node):
    """
    A leaf node that stores the final prediction value.

    Attributes:
        value (any): Predicted value at this leaf.
        depth (int): Depth of this leaf in the tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf (no further children).
        """
        return self.depth


class Decision_Tree:
    """
    Main class to build and manage the decision tree.

    Attributes:
        max_depth (int): Maximum depth allowed for the tree.
        min_pop (int): Minimum number of samples to allow a split.
        seed (int): Random seed for reproducibility.
        split_criterion (str): Method used to choose splits (e.g., 'random').
        root (Node): Root node of the tree.
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
        Returns the overall depth of the tree from the root.
        """
        return self.root.max_depth_below()
