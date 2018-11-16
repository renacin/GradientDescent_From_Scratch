# Name:                                            Renacin Matadeen
# Student Number:                                        N/A
# Date:                                              09/30/2018
# Course:                                                N/A
# Title                                          Quadratic Regression
#
#
#
#
#
# ----------------------------------------------------------------------------------------------------------------------

import time
import numpy as np
import random
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
"""

In many analyses, multiple variables make up the basis of the experiment. As a result if would be beneficial to look at
an implementation of Multivariate-Linear Regression. Note that this experiment will use Mini-Batch Stochastic Gradient
Descent, specifically within the context of identifying Miles Per Gallon given, Weight, and Horsepower.

From the tests conducted, a small learning rate may be necessary in some cases. Note that this can change quite
significantly when we rate our data!

"""
# ----------------------------------------------------------------------------------------------------------------------


def gradient_descent(x1, x2, y):
    # Number Of Iterations, Learning Rate, & Batch Size
    iterations = 100000
    learning_rate = 0.5
    n = 50

    # Time Functions
    start_time = time.time()

    # Start With Some Random Value Of M1, M2, and B
    m1_curr = random.randint(0, 100)
    m2_curr = random.randint(0, 100)
    b_curr = random.randint(0, 100)

    # Gather Data
    cost_values = []
    m1_values = []
    m2_values = []
    b_values = []

    # Print Heading
    print("{:-^55}".format("Gradient Descent"))

    # Iteration
    for i in range(iterations):

        # Get Random Points In Dataset, Takes Batch Size
        x1_batch = []
        x2_batch = []
        y_batch = []

        # Can I Use String Formatting?
        for num in range(0, n):
            r_int = int(random.randint(1, x1.size))
            x1_batch.append(x1.item(r_int - 1))
            x2_batch.append(x2.item(r_int - 1))
            y_batch.append(y.item(r_int - 1))

        # Turn To Np.Arrays
        x1_batch = np.array(x1_batch)
        x2_batch = np.array(x2_batch)
        y_batch = np.array(y_batch)

        # Find The Y Predicted Given The m & b
        y_predicted = (m1_curr * x1_batch) + (m2_curr * x2_batch) + b_curr

        # Find The Cost Of The Current Model, Using List Comprehension, Works Because Of Np.Array, Not Python List
        # Y & Y_Predicted Are Lists, Np.Array Allows For Matrix Subtraction
        cost = (1/n) * sum([val**2 for val in (y_batch - y_predicted)])

        # Find The m1, m2 & b derivative
        md1 = -(2 / n) * sum(x1_batch * (y_batch - y_predicted))
        md2 = -(2 / n) * sum(x2_batch * (y_batch - y_predicted))
        bd = -(2 / n) * sum(y_batch - y_predicted)

        # Adjust m & b values. Learning Rate Is Subtracted Because Descent In Gradient Descent
        m1_curr = m1_curr - learning_rate * md1
        m2_curr = m2_curr - learning_rate * md2
        b_curr = b_curr - learning_rate * bd

        # Print Values
        print("Iteration {0} | Cost: {1:.5f} | M1: {2:.5f} | M2: {3:.5f} | B: {4:.5f}"
              .format(i + 1, cost, m1_curr, m2_curr, b_curr))

        cost_values.append(cost)
        m1_values.append(m1_curr)
        m2_values.append(m2_curr)
        b_values.append(b_curr)

    # End Timer
    total_time = time.time() - start_time
    print("\n{:-^55}".format("Estimated Time"))
    print("Seconds: {0:.2f}".format(total_time))

    return m1_curr, m2_curr, b_curr, cost_values, m1_values, m2_values, b_values

# ----------------------------------------------------------------------------------------------------------------------


# Create Entry Point
if __name__ == "__main__":

    # Read Data In CSV File
    data_dir = "/Users/Matadeen/Documents/Programming/Python/Gradient_Descent/Tests/Auto_Test_Rated.csv"
    df = pd.read_csv(data_dir)

    # Clean Data
    df = df.fillna(0)

    # Turn Data To Lists
    weight = np.array(df['R_Wght'].values.tolist())
    horsepower = np.array(df['R_Hrsp'].values.tolist())
    mpg = np.array(df['R_MPG'].values.tolist())

    # Instantiate GD Function, And Collect Data
    m1, m2, b, cost_values, m1_values, m2_values, b_values = gradient_descent(weight, horsepower, mpg)

    # Plots

    # Instantiate 3D Plot
    fig = plt.figure(figsize=plt.figaspect(0.50))

    # Plot Of Data - 3D | SubPlot 1|
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(df["R_Wght"], df["R_Hrsp"], df["R_MPG"], marker="+", s=1, color="red")
    plt.title('Dependencies Of MPG')
    ax.set_xlabel('R_Wght')
    ax.set_ylabel('R_Hrsp')
    ax.set_zlabel('R_MPG')

    # Plot Of M1, M2, and B Values - 3D | SubPlot 2|
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.scatter(m1_values, m2_values, b_values)
    plt.title('Parameter Iteration')
    ax1.set_xlabel('M1')
    ax1.set_ylabel('M2')
    ax1.set_zlabel('Bias')
    plt.show()

    # Show Cost Over Iterations
    plt.xlabel('Iteration')
    plt.ylabel('Iteration Cost')
    plt.title('Cost Values Over Iterations')
    plt.grid(True)
    iteration = [x for x in range(len(cost_values))]

    # Skip The First 100 Iterations, Too Much Noise
    skp_sz = 100
    plt.plot(iteration[skp_sz:], cost_values[skp_sz:])
    plt.show()

    print("\n{:-^55}".format("Estimations"))

    pred_wght = [3504, 2672, 4464]
    pred_hpwr = [130, 110, 175]

    for val in zip(pred_wght, pred_hpwr):
        pred_mpg = (m1 * val[0]) + (m2 * val[1]) + b
        print("If Weight = {0:.2f} & Horsepower = {1:.2f}, Then MPG = {2:.2f}".format(val[0], val[1], pred_mpg))





