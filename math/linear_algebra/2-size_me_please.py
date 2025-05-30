#!/usr/bin/env python3
"""
Module that defines the matrix_shape function.
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list: A list of integers representing the shape.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
