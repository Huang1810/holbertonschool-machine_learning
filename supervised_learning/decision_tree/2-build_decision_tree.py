#!/usr/bin/env python3
"""
Defines basic classes for building a decision tree: Node, Leaf, and Decision_Tree.
"""

import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.

    Attributes:
        feature (int): Index of the feature to split on.
        threshold (float): Threshold value for the split.
        left_child (Node): Left subtree.
        right_child (Node): Right subtree.
        is_root (bool): Whether the node is the root.
        depth (int): Depth of the node in the tree.
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
        Returns a formatted string representation of the subtree rooted at this node.
        """
        prefix = "root" if self.is_root else "-> node"
        result = f"{prefix} [feature={self.feature}, threshold={self.threshold}]"
        if self.left_child:
            result += "\n" + left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += "\n" + right_child_add_prefix(str(self.right_child))
        return result

    def max_depth_below(self):
        """
        Returns the maximum depth of the subtree rooted at this node.
        """
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts nodes in the subtree rooted at this node.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves.
        """
        if only_leaves:
            return int(self.is_leaf) + sum([
                child.count_nodes_below(only_leaves)
                for child in [self.left_child, self.right_child] if child
            ])
        count = 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count


class Leaf:
    """
    Represents a terminal (leaf) node in the decision tree.

    Attributes:
        value (any): Predicted value at the leaf.
        depth (int): Depth of the leaf in the tree.
    """
    def __init__(self, value, depth=None):
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Returns a formatted string representation of the leaf.
        """
        return f"-> leaf [value={self.value}]"

    def max_depth_below(self):
        """
        Returns the depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Always returns 1, as this is a single leaf node.
        """
        return 1


class Decision_Tree:
    """
    Basic decision tree structure.

    Attributes:
        max_depth (int): Maximum allowed depth of the tree.
        min_pop (int): Minimum number of samples to allow a split.
        seed (int): Seed for reproducibility.
        split_criterion (str): Criterion used for splitting nodes.
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

    def __str__(self):
        """
        Returns a string representation of the entire tree.
        """
        return self.root.__str__()

    def depth(self):
        """
        Returns the maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts total nodes or only leaf nodes.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Total number of nodes or leaf nodes.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)


def left_child_add_prefix(text):
    """
    Formats the left child's string representation with indentation.

    Args:
        text (str): String representation of the child node.

    Returns:
        str: Formatted string with tree-like indentation.
    """
    lines = text.split("\n")
    new_text = "    +---" + lines[0] + "\n"
    for x in lines[1:]:
        if x.strip():  # Only add prefix to non-empty lines
            new_text += ("    |  " + x) + "\n"
        else:
            new_text += "\n"
    return new_text.rstrip("\n")


def right_child_add_prefix(text):
    """
    Formats the right child's string representation with indentation.

    Args:
        text (str): String representation of the child node.

    Returns:
        str: Formatted string with tree-like indentation.
    """
    lines = text.split("\n")
    new_text = "    +---" + lines[0] + "\n"
    for x in lines[1:]:
        if x.strip():  # Only add prefix to non-empty lines
            new_text += ("       " + x) + "\n"
        else:
            new_text += "\n"
    return new_text.rstrip("\n")
