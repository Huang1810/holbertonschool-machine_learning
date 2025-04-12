#!/usr/bin/env python3

"""
Module 9-sum_total
This module provides a function to calculate the sum of squares of integers
from 1 to n.
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to n.

    Args:
        n (int): The stopping condition for the summation.

    Returns:
        int: The sum of squares from 1 to n if n is a valid positive integer.
             Otherwise, return None.
    """

    if not isinstance(n, int) or n <= 0:
        return None

    return (n * (n + 1) * (2 * n + 1)) // 6
