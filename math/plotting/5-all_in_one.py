#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def all_in_one():

    # Data for the first plot (y0 = x^3)
    y0 = np.arange(0, 11) ** 3

    # Data for the second plot (Men's height vs weight)
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    # Data for the third plot (Exponential decay of C-14)
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    # Data for the fourth plot (Exponential decay of radioactive elements)
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    # Data for the fifth plot (Student grades)
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Create a figure with a 3x2 grid
    plt.figure(figsize=(12, 10))
    plt.suptitle('All in One', fontsize='x-small')

    # First plot (y = x^3)
    plt.subplot(3, 2, 1)
    plt.plot(y0, 'b-')
    plt.xlabel('X', fontsize='x-small')
    plt.ylabel('y', fontsize='x-small')
    plt.title('Plot of y = x^3', fontsize='x-small')

    # Second plot (Height vs Weight)
    plt.subplot(3, 2, 2)
    plt.scatter(x1, y1, c='magenta')
    plt.xlabel('Height (in)', fontsize='x-small')
    plt.ylabel('Weight (lbs)', fontsize='x-small')
    plt.title("Men's Height vs Weight", fontsize='x-small')

    # Third plot (Exponential decay of C-14)
    plt.subplot(3, 2, 3)
    plt.plot(x2, y2, 'r-')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of C-14', fontsize='x-small')
    plt.yscale('log')

    # Fourth plot (Exponential decay of radioactive elements)
    plt.subplot(3, 2, 4)
    plt.plot(x3, y31, 'r--', label='C-14')
    plt.plot(x3, y32, 'g-', label='Ra-226')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    plt.legend(loc='upper right', fontsize='x-small')

    # Fifth plot (Student grades histogram)
    plt.subplot(3, 2, 5)
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.title('Project A', fontsize='x-small')
    plt.xticks(range(0, 101, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    # Sixth plot (Empty, spanning 2 columns)
    plt.subplot(3, 2, (6, 7))
    plt.axis('off')

    # Display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
