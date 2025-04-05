#!/usr/bin/env python3
"""Module to perform matrix multiplication without using loops"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication of two NumPy matrices.

    Args:
        mat1: The first matrix.
        mat2: The second matrix.

    Returns:
        A new matrix resulting from the matrix multiplication of mat1 and mat2.
    """
    return np.dot(mat1, mat2)
