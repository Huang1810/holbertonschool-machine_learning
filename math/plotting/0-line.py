#!/usr/bin/env python3
"""
This module contains a function to plot a line graph of y = x^3.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots y = x^3 as a solid red line.

    The x-axis ranges from 0 to 10.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, color='red', linestyle='-')

    plt.xlim(0, 10)

    plt.show()
