#!/usr/bin/env python3
"""
This module provides a function to transpose a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Return the transpose of a 2D matrix.

    Args:
        matrix (list of lists): The original 2D matrix.

    Returns:
        list of lists: The transposed matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
