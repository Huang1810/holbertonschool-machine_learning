#!/usr/bin/env python3
"""Module for NumPy matrix concatenation"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two NumPy matrices along a specified axis.

    Args:
        mat1: First matrix to concatenate.
        mat2: Second matrix to concatenate.
        axis: Axis along which to perform the concatenation.

    Returns:
        New concatenated matrix of mat1 and mat2.
    """
    return np.concatenate((mat1, mat2), axis=axis)