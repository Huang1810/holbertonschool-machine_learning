#!/usr/bin/env python3
"""
Decision Tree Classifier: Defines classes for building a basic decision tree,
including internal nodes, leaf nodes, and the overall tree structure.
"""

import numpy as np


class Node:
    """
    Represents an internal node in the decision tree.
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
        Returns the maximum depth of the subtree rooted at this node.
        """
        left_child_depth = self.left_child.max_depth_below()
        right_child_depth = self.right_child.max_depth_below()
        return max(left_child_depth, right_child_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts nodes in the subtree rooted at this node.
        """
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )
        return (left_count + right_count if only_leaves
                else 1 + left_count + right_count)

    def __str__(self):
        """
        Returns a formatted string representation of the subtree.
        """
        if self.is_root:
            node_str = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            node_str = f"-> node [feature={self.feature}, threshold={self.threshold}]"

        left_str = str(self.left_child)
        right_str = str(self.right_child)

        left_formatted = self.left_child_add_prefix(left_str)
        right_formatted = self.right_child_add_prefix(right_str)

        result = node_str
        if left_formatted:
            result += "\n" + left_formatted
        if right_formatted:
            result += right_formatted
        return result

    def left_child_add_prefix(self, text):
        """
        Adds formatting prefix to the left child's string for pretty printing.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |  " + line + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds formatting prefix to the right child's string for pretty printing.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    " + line + "\n"
        return new_text.rstrip("\n")

    def get_leaves_below(self):
        """
        Recursively collects all leaves under this node.
        """
        return (
            self.left_child.get_leaves_below() +
            self.right_child.get_leaves_below()
        )


class Leaf(Node):
    """
    Represents a leaf (terminal) node in the decision tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of this leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 since this is a leaf node.
        """
        return 1

    def __str__(self):
        """
        Returns a formatted string for this leaf node.
        """
        return (
            f"-> leaf [value={self.value}]"
        )

    def get_leaves_below(self):
        """
        Returns the leaf itself as it's terminal.
        """
        return [self]


class Decision_Tree:
    """
    Decision Tree classifier class.
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
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
        Returns the maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the full tree.
        """
        return str(self.root)

    def get_leaves(self):
        """
        Returns all the leaf nodes in the tree.
        """
        return self.root.get_leaves_below()
