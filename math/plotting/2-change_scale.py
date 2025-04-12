#!/usr/bin/env python3
"""This module plots exponential decay of C-14 with a logarithmic y-axis."""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plot the exponential decay of Carbon-14.

    This function uses the half-life of Carbon-14 to compute and plot
    the fraction of Carbon-14 remaining over time, on a logarithmic scale.
    The x-axis represents time in years, and the y-axis shows the
    fraction remaining on a logarithmic scale.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y)
    plt.yscale('log')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')
    plt.xlim(0, 28650)

    plt.show()
