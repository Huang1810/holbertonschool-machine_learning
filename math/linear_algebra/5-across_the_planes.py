#!/usr/bin/env python3
"""
This module provides a function to add two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): First matrix.
        mat2 (list of list of int/float): Second matrix.

    Returns:
        list or None: New matrix with element-wise sums or None
        if shapes differ.
    """
    if len(mat1) != len(mat2) or any(
        len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)
    ):
        return None
    return [
        [a + b for a, b in zip(row1, row2)]
        for row1, row2 in zip(mat1, mat2)
    ]
