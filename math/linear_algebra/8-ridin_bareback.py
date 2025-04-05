#!/usr/bin/env python3


"""
    Multiplies two 2D matrices.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices mat1 and mat2.

    Args:
        mat1 (list): First 2D matrix (list of lists of ints/floats).
        mat2 (list): Second 2D matrix (list of lists of ints/floats).

    Returns:
        list: A new 2D matrix resulting from multiplying mat1 and mat2,
              or None if the matrices cannot be multiplied.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = [[0] * len(mat2[0]) for _ in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
