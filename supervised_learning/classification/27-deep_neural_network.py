#!/usr/bin/env python3

"""
This is the DeepNeuralNetwork class module.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    Represents a deep neural network to perform classification.
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network object.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:

                prev_layer = nx
            else:

                prev_layer = layers[i - 1]

            self.__weights[f"W{i + 1}"] = np.random.randn(
                    layers[i], prev_layer) * np.sqrt(2 / prev_layer)

            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        """

        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):

            prev_A = self.__cache[f"A{i - 1}"]

            Z = np.matmul(self.__weights[f"W{i}"], prev_A)\
                + self.__weights[f"b{i}"]

            if i < self.__L:
                activation = 1 / (1 + np.exp(-Z))
            else:

                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                activation = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

            self.__cache[f"A{i}"] = activation

        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        return -(1 / m) * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """
        Evaluates the neural netowrk's predictions
        """
        output, cache = self.forward_prop(X)

        prediction = np.eye(output.shape[0])[np.argmax(output, axis=0)].T

        cost = self.cost(Y, output)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the network.
        """
        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y

        for i in range(self.__L, 0, -1):

            prev_A = cache[f"A{i - 1}"]

            dW = (1 / m) * np.matmul(dZ, prev_A.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            dZ = np.matmul(
                    self.__weights[f"W{i}"].T, dZ) * prev_A * (1 - prev_A)

            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Method to train deep neural network
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        count = []

        for i in range(iterations + 1):

            A, cache = self.forward_prop(X)

            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            cost = self.cost(Y, A)

            costs.append(cost)
            count.append(i)

            if verbose and (i % step == 0 or i == 0 or i == iterations):

                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetowrk object.
        """

        try:
            with open(filename, "rb") as file:
                unpickled_obj = pickle.load(file)
            return unpickled_obj

        except FileNotFoundError:
            return None
