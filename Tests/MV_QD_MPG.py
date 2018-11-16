# Name:                                            Renacin Matadeen
# Student Number:                                        N/A
# Date:                                              10/03/2018
# Course:                                                N/A
# Title                                     Multivariate Quadratic Regression
#
#
#
#
#
# ----------------------------------------------------------------------------------------------------------------------

import time
import numpy as np
import pandas as pd
import random
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


def gradient_descent(x1, x2, y, batch, iterations, learning_rate):
    # Time Functions
    start_time = time.time()

    # Start With Some Value For Ax1, Bx1, Ax2, Bx2, and Cx
    ax1_curr = random.randint(0, 100)
    bx1_curr = random.randint(0, 100)
    ax2_curr = random.randint(0, 100)
    bx2_curr = random.randint(0, 100)
    cx_curr = random.randint(0, 100)

    # Print Heading
    print("{:-^55}".format("Gradient Descent"))

    # Gather Data
    cost_values = []
    ax1_values = []
    bx1_values = []
    ax2_values = []
    bx2_values = []
    cx_values = []

    # Batch Size
    n = batch

    # Iteration
    for i in range(iterations):

        # Get Random Points In Dataset, Takes Batch Size
        x1_batch = []
        x2_batch = []
        y_batch = []

        for num in range(0, n):
            r_int = int(random.randint(1, x1.size))
            x1_batch.append(x1.item(r_int - 1))
            x2_batch.append(x2.item(r_int - 1))
            y_batch.append(y.item(r_int - 1))

        # Turn To Np.Arrays
        x1_batch = np.array(x1_batch)
        x2_batch = np.array(x2_batch)
        y_batch = np.array(y_batch)

        # Find The Y Predicted Given The Equation F(x, y) = (Ax^2 + Bx) + (Ay^2 + By) + C
        y_predicted = ((ax1_curr * (x1_batch ** 2)) + (bx1_curr * x1_batch)) + \
                      ((ax2_curr * (x2_batch ** 2)) + (bx2_curr * x2_batch)) + cx_curr

        # Find The Cost Of The Current Model
        cost = (1 / n) * sum([val ** 2 for val in (y_batch - y_predicted)])

        # Find The Derivative Of Parameter A, B, And C, Given Equation AX^2 + BX + C
        ax1_md = -(2 / n) * sum((x1_batch ** 2) * (y_batch - y_predicted))
        bx1_md = -(2 / n) * sum(x1_batch * (y_batch - y_predicted))

        ax2_md = -(2 / n) * sum((x2_batch ** 2) * (y_batch - y_predicted))
        bx2_md = -(2 / n) * sum(x2_batch * (y_batch - y_predicted))

        cx_md = -(2 / n) * sum(y_batch - y_predicted)

        # Adjust m & b values. Learning Rate Is Subtracted Because Descent In Gradient Descent
        ax1_curr = ax1_curr - learning_rate * ax1_md
        bx1_curr = bx1_curr - learning_rate * bx1_md

        ax2_curr = ax2_curr - learning_rate * ax2_md
        bx2_curr = bx2_curr - learning_rate * bx2_md

        cx_curr = cx_curr - learning_rate * cx_md

        # Print Values
        print("Iteration {0} | Cost: {1:.3f} | A1: {2:.3f} | B1: {3:.3f} | A2: {4:.3f} | B2: {5:.3f} | C: {6:.3f}"
              .format(i + 1, cost, ax1_curr, bx1_curr, ax2_curr, bx2_curr, cx_curr))

        cost_values.append(cost)
        ax1_values.append(ax1_curr)
        bx1_values.append(bx1_curr)
        ax2_values.append(ax2_curr)
        bx2_values.append(bx2_curr)
        cx_values.append(cx_curr)

    # End Timer
    total_time = time.time() - start_time
    print("\n{:-^55}".format("Estimated Time"))
    print("Seconds: {0:.2f}".format(total_time))

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

    # Define Hyper-parameters
    batch = 20
    iterations = 100000
    learning_rate = 0.05

    # Instantiate GD Function, And Collect Data
    gradient_descent(weight, horsepower, mpg, batch, iterations, learning_rate)
