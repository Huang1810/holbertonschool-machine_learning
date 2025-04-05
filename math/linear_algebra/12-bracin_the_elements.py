#!/usr/bin/env python3
"""Module to perform element-wise operations on two NumPy matrices"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division
    on two NumPy matrices.

    Args:
        mat1: First matrix
        mat2: Second matrix

    Returns:
        tuple: Contains the results of the element-wise addition, subtraction,
               multiplication, and division.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
