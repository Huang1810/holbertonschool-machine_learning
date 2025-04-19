#!/usr/bin/env python3
"""
This module defines classes for constructing a basic decision tree,
including Node, Leaf, and Decision_Tree.
"""
import numpy as np


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (int): Index of the feature used to split the data.
        threshold (float): Threshold value for the split.
        left_child (Node): Left child node.
        right_child (Node): Right child node.
        is_root (bool): Indicates whether the node is the root.
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
        Returns a string representation of the node and its children.

        Returns:
            str: The string representation of the node and its children.
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
        Adds indentation to the left child's subtree representation.

        Args:
            text (str): The string representation of the left child's subtree.

        Returns:
            str: The modified string with added prefixes.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds indentation to the right child's subtree representation.

        Args:
            text (str): The string representation of the right child's subtree.

        Returns:
            str: The modified string with added prefixes.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "       " + x + "\n"
        return new_text

    def max_depth_below(self):
        """
        Computes the maximum depth of the tree from this node downward.

        Returns:
            int: The maximum depth of the tree below this node.
        """
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the nodes below this node, optionally counting only the leaf nodes.

        Args:
            only_leaves (bool): Whether to count only leaf nodes.

        Returns:
            int: The count of nodes or leaves below this node.
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
        Retrieves all leaf nodes beneath this node.

        Returns:
            list: A list of all leaf nodes below this node.
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
        Recursively updates the bounds for this node and its children.
        Initializes with infinite bounds at the root and adjusts based on data.
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
            child.update_bounds_below()

    def update_indicator(self):
        """
        Updates the indicator function for the node based on bounds.
        This function evaluates whether a feature vector satisfies the node's conditions.
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
        Updates the prediction function for the tree by computing bounds,
        retrieving leaves, and setting their indicators.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([leaf.value
                                           for x in A for leaf in leaves
                                           if leaf.indicator(x)])

    def pred(self, x):
        """
        Recursively predicts the value by traversing the tree based on input features.

        Args:
            x (array): The feature vector for a single sample.

        Returns:
            any: The predicted value for the input sample.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Represents a leaf node in the decision tree.

    Attributes:
        value (any): The predicted value at the leaf.
        depth (int): Depth of the leaf in the tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Returns a string representation of the leaf.

        Returns:
            str: The string representation of the leaf.
        """
        return f"-> leaf [value={self.value}] "

    def max_depth_below(self):
        """
        Returns the depth of the leaf, which is the end of a branch.

        Returns:
            int: The depth of this leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns the count of this node as a leaf.

        Args:
            only_leaves (bool): Ignored because a leaf always counts as one.

        Returns:
            int: Always 1, since a leaf is a single node.
        """
        return 1

    def get_leaves_below(self):
        """
        Returns the leaf node itself in a list.

        Returns:
            list: A list containing only this leaf node.
        """
        return [self]

    def update_bounds_below(self):
        """
        Leaves do not update bounds, so this is a no-op.
        """
        pass

    def pred(self, x):
        """
        Returns the predicted value for this leaf based on its stored value.

        Args:
            x (array): The feature vector for a single sample.

        Returns:
            any: The predicted value.
        """
        return self.value


class Decision_Tree():
    """
    Represents a decision tree.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_pop (int): Minimum population required for splitting nodes.
        seed (int): Random seed for initialization.
        split_criterion (str): Criterion used for node splitting.
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

    def __str__(self):
        """
        Returns a string representation of the entire decision tree.

        Returns:
            str: A string representation of the decision tree.
        """
        return self.root.__str__()

    def depth(self):
        """
        Returns the maximum depth of the tree.

        Returns:
            int: The maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the nodes in the tree, optionally counting only leaf nodes.

        Args:
            only_leaves (bool): Whether to count only leaf nodes.

        Returns:
            int: Total number of nodes or leaf nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """
        Returns all leaves in the tree.

        Returns:
            list: A list of all leaf nodes in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Recursively updates bounds for all nodes starting from the root.
        """
        self.root.update_bounds_below()

    def update_indicator(self):
        """
        Updates the indicator functions for all nodes in the tree starting
        from the root.
        """
        self.root.update_indicator()

    def pred(self, x):
        """
        Makes a prediction based on the input features by delegating to the root node.

        Args:
            x (array): The feature vector for a sample.

        Returns:
            any: The predicted value from the tree.
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        Updates the prediction function for the tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([next(leaf.value for leaf in leaves
                                                 if leaf.indicator(x.reshape(1, -1)))
                                          for x in np.atleast_2d(A)])
