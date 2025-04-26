#!/usr/bin/python3
"""Defines minOperations function."""

def minOperations(n):
    """
    Calculates the minimum number of operations needed to get exactly n 'H' characters.

    Operations allowed:
    - Copy All
    - Paste

    Args:
        n (int): The target number of 'H' characters.

    Returns:
        int: The minimum number of operations, or 0 if impossible.
    """
    if n <= 1:
        return 0

    operations = 0
    divisor = 2

    while n > 1:
        while n % divisor == 0:
            operations += divisor
            n //= divisor
        divisor += 1

    return operations
