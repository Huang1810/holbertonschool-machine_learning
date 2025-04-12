#!/usr/bin/env python3

"""
Module 10-matisse
This module provides a function to calculate the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly (list of int): A list of coefficients representing a polynomial.

    Returns:
        list: A list of coefficients representing the derivative of the polynomial.
              Returns [0] if the derivative is 0.
              Returns None if the input is invalid.
    """

    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power, coef in enumerate(poly):
        if power > 0:
            derivative.append(coef * power)

    if not derivative:
        return [0]

    return derivative
