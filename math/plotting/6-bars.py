#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    # Seed the random number generator for reproducibility
    np.random.seed(5)

    # Fruit matrix representing the number of apples, bananas, oranges, and peaches
    fruit = np.random.randint(0, 20, (4, 3))  # 4 rows (fruit types), 3 columns (people)

    # Define the colors for each fruit
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']  # apples, bananas, oranges, peaches

    # Define the labels for the x-axis (people's names)
    people = ['Farrah', 'Fred', 'Felicia']

    # Create a figure
    plt.figure(figsize=(6.4, 4.8))

    # Create a stacked bar plot with a bar width of 0.5
    plt.bar(people, fruit[0], color=colors[0], label='Apples', width=0.5)  # Apples
    plt.bar(people, fruit[1], bottom=fruit[0], color=colors[1], label='Bananas', width=0.5)  # Bananas
    plt.bar(people, fruit[2], bottom=fruit[0] + fruit[1], color=colors[2], label='Oranges', width=0.5)  # Oranges
    plt.bar(people, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2], color=colors[3], label='Peaches', width=0.5)  # Peaches

    # Set the labels and title
    plt.xlabel('People', fontsize='x-small')
    plt.ylabel('Quantity of Fruit', fontsize='x-small')
    plt.title('Number of Fruit per Person', fontsize='x-small')

    # Set the y-axis limits and ticks
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10), fontsize='x-small')

    # Add a legend
    plt.legend(title='Fruit Types', fontsize='x-small')

    # Show the plot
    plt.tight_layout()
    plt.show()
