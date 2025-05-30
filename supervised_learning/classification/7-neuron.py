#!/usr/bin/env python3
"""
Wrote a class Neuron that defines a single neuron performing binary
classification with enhanced training capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Class that defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Constructor for the neuron class.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)

        self.__b = 0

        self.__A = 0

    @property
    def W(self):
        """
        Getter for the private attribute __W.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the private attribute __b.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the private attribute __A
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.
        """

        Z = np.dot(self.__W, X) + self.__b

        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.
        """
        A = self.forward_prop(X)

        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent on the neuron
        """
        m = Y.shape[1]

        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron using gradient descent to minimize the cost
        function over multiple iterations.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")


        costs = []
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(range(0, iterations + 1, step), costs, 'b')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
