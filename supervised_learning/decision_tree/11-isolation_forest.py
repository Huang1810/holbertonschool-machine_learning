#!/usr/bin/env python3
"""
Module implementing the Isolation_Random_Forest class for anomaly detection
using Isolation Trees.
Designed to handle high-dimensional datasets, it identifies anomalies by
isolating data points through random feature splits.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    A class that implements the Isolation Forest algorithm for anomaly detection.
    An Isolation Forest consists of multiple Isolation Random Trees that
    isolate observations by randomly selecting features and split values 
    based on the maximum and minimum values of the selected feature.

    Attributes:
        n_trees (int): Number of trees in the forest.
        max_depth (int): The maximum depth allowed for each tree in the forest.
        min_pop (int): Minimum sample size in a node to continue splitting.
        seed (int): Random seed used for reproducibility.
        numpy_preds (list): List of prediction functions from each tree in the forest.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the Isolation Random Forest with the specified parameters.

        Args:
            n_trees (int): Number of trees in the forest.
            max_depth (int): Maximum depth of each tree in the forest.
            min_pop (int): Minimum sample size in a node to consider further splitting.
            seed (int): Seed for the random number generator to ensure reproducibility.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the anomaly scores for each sample in the dataset.

        Args:
            explanatory (numpy.ndarray): Data for which anomaly scores are predicted.

        Returns:
            numpy.ndarray: Averaged depth of each sample across all trees in the forest.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Fits the Isolation Forest model to the provided data by training multiple
        Isolation Random Trees.

        Args:
            explanatory (numpy.ndarray): The dataset used to fit the model.
            n_trees (int, optional): Number of trees in the forest (default is 100).
            verbose (int, optional): Verbosity level; 0 is silent, 1 prints tree statistics after training.

        Returns:
            None
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }""")

    def suspects(self, explanatory, n_suspects):
        """
        Identifies the top n suspects with the lowest mean depth across all trees,
        suggesting they are potential outliers.

        Args:
            explanatory (numpy.ndarray): The explanatory variables of the dataset.
            n_suspects (int): The number of suspect data points to return.

        Returns:
            tuple: Two numpy arrays; the first contains the suspect data points,
                and the second contains the corresponding depths indicating their isolation levels.
        """
        # Calculate the mean depth for each data point using the predict method
        depths = self.predict(explanatory)
        # Get the indices that would sort the depths array in ascending order
        sorted_indices = np.argsort(depths)
        # Select the top n suspects with the smallest depths
        suspect_data = explanatory[sorted_indices[:n_suspects]]
        suspect_depths = depths[sorted_indices[:n_suspects]]
        return suspect_data, suspect_depths
