#!/usr/bin/env python3
"""
This module defines the classes for constructing a basic decision tree,
including Node, Leaf, and Decision_Tree.
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

    def __str__(self):
        """
        Returns a visual representation of the subtree rooted at this node.
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
        Prefixes the left child subtree for visualization.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Prefixes the right child subtree for visualization.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "       " + x + "\n"
        return new_text

    def max_depth_below(self):
        """
        Computes the maximum depth from this node downward.
        """
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes or only leaves below this node.
        """
        if only_leaves:
            return 1 if self.is_leaf else 0
        count = 1  # Count current node
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def get_leaves_below(self):
        """
        Returns a list of all leaves under this node.
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
        Updates upper/lower bounds recursively for all descendant nodes.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()


class Leaf(Node):
    """
    Represents a leaf node in the tree with a predicted value.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Returns a string representation of the leaf.
        """
        return f"-> leaf [value={self.value}]"

    def max_depth_below(self):
        """
        Returns the depth of this leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Always returns 1, since this is a single leaf.
        """
        return 1

    def get_leaves_below(self):
        """
        Returns a list containing only this leaf.
        """
        return [self]

    def update_bounds_below(self):
        """
        Placeholder for bounds update. No action needed for leaves.
        """
        pass


class Decision_Tree:
    """
    Represents the decision tree and provides methods to analyze its structure.
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
        Counts all nodes or only leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """
        Returns a list of all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Starts recursive bounds update from the root.
        """
        self.root.update_bounds_below()
