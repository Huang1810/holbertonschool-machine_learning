#!/usr/bin/env python3
"""
This module provides a function to concatenate two 2D matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specified axis.

    Args:
        mat1 (list): First 2D matrix (list of lists of ints/floats).
        mat2 (list): Second 2D matrix (list of lists of ints/floats).
        axis (int): The axis along which to concatenate (0 for rows, 1 for columns).

    Returns:
        list: A new 2D matrix resulting from concatenating mat1 and mat2 along the specified axis.
              If the matrices cannot be concatenated, returns None.
    """
    if axis == 0:

        if len(mat1[0]) == len(mat2[0]):
            return mat1 + mat2
        return None
    elif axis == 1:

        if len(mat1) == len(mat2):
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        return None
    return None
