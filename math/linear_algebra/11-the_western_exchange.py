#!/usr/bin/env python3
"""Module for transposing a NumPy matrix"""


def np_transpose(matrix):
    """
    Transposes a NumPy array.

    Args:
        matrix (numpy.ndarray): The matrix to transpose.

    Returns:
        numpy.ndarray: The transposed matrix.
    """
    return matrix.transpose()
