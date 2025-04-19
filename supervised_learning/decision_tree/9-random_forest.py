#!/usr/bin/env python3
"""
This module implements a Random Forest classifier using Decision Trees
as base learners. It provides methods for fitting the model to training
data, making predictions, and evaluating accuracy on test data.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """
    Random Forest classifier that uses Decision Trees as base learners.

    This class implements a Random Forest classifier by constructing multiple
    Decision Trees and using them to make predictions through a majority vote.
    It supports training with specified parameters, prediction on test data,
    and evaluation of the model's accuracy.

    Attributes:
        n_trees (int): The number of trees in the forest (default is 100).
        max_depth (int): The maximum depth of each Decision Tree (default is 10).
        min_pop (int): The minimum number of samples required to split a node (default is 1).
        seed (int): The random seed used for initializing each tree.
        numpy_preds (list): A list of prediction functions from the trained Decision Trees.
        target (numpy.ndarray): The target values for training.
        explanatory (numpy.ndarray): The feature values for training.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the Random Forest classifier.

        Args:
            n_trees (int): The number of trees in the forest (default is 100).
            max_depth (int): The maximum depth of the trees (default is 10).
            min_pop (int): The minimum samples required to split a node (default is 1).
            seed (int): The random seed for tree initialization (default is 0).
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Makes predictions for the target variable based on input features.

        Args:
            explanatory (numpy.ndarray): The input features for prediction.

        Returns:
            numpy.ndarray: The predicted target values based on majority voting
                           from all the decision trees.
        """
        predictions = []

        # Generate predictions from each tree in the forest
        for predict_function in self.numpy_preds:
            predictions.append(predict_function(explanatory))

        predictions = np.array(predictions)

        # Calculate the mode (most frequent) prediction for each instance
        mode_predictions = []
        for example_predictions in predictions.T:
            unique_values, counts = np.unique(example_predictions,
                                              return_counts=True)
            mode_index = np.argmax(counts)
            mode_predictions.append(unique_values[mode_index])

        return np.array(mode_predictions)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Trains the Random Forest classifier using the provided data.

        Args:
            explanatory (numpy.ndarray): The feature matrix for training.
            target (numpy.ndarray): The target variable for training.
            n_trees (int): The number of trees in the forest (default is 100).
            verbose (int): Verbosity level (0 for silent, 1 for detailed output).

        Returns:
            None
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        
        # Build and train each decision tree
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,
                              seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        # Print training statistics if verbose is enabled
        if verbose == 1:
            print(f"Training finished.")
            print(f"  - Mean depth                     : {np.array(depths).mean()}")
            print(f"  - Mean number of nodes           : {np.array(nodes).mean()}")
            print(f"  - Mean number of leaves          : {np.array(leaves).mean()}")
            print(f"  - Mean accuracy on training data : {np.array(accuracies).mean()}")
            print(f"  - Accuracy of the forest on td   : {self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """
        Evaluates the accuracy of the Random Forest on the test data.

        Args:
            test_explanatory (numpy.ndarray): The feature matrix for testing.
            test_target (numpy.ndarray): The true target values for testing.

        Returns:
            float: The accuracy of the model on the test data, calculated as
                   the proportion of correct predictions.
        """
        correct_predictions = np.sum(np.equal(self.predict(test_explanatory),
                                              test_target))
        return correct_predictions / test_target.size
