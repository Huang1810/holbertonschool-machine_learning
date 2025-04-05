#!/usr/bin/env python3
"""
This module provides a function to concatenate two arrays (lists).
"""


def cat_arrays(arr1, arr2):
    """
    Concatenates two arrays (lists) and returns a new list.

    Args:
        arr1 (list): First array (list of ints/floats).
        arr2 (list): Second array (list of ints/floats).

    Returns:
        list: A new list containing all elements of arr1 followed by arr2.
    """
    return arr1 + arr2
