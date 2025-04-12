#!/usr/bin/env python3
"""This module plots a histogram of student grades for Project A."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot a histogram of student scores with bins every 10 units."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    bins = np.arange(0, 101, 10)  # bins every 10 units from 0 to 100
    plt.hist(student_grades, bins=bins, edgecolor='black')

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    plt.show()
