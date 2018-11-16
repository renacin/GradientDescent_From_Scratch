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
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
"""

Can we implement all that we have learned, and develop a quadratic regression model? Yes we can. Once we derive 
the partial derivative for each constant - with respect to the Mean Square Error (MSE), in this case, A, B, and C we can
quite easily implement gradient descent to find the optimal parameters. 

Remember:
    The Equation For A Quadratic Formula Is
        F(X) = Ax^2 + BX + C
        
    The Equation For Multivariate Quadratic Regression Is
         F(X) = (Ax^2 + Bx) + (Ay^2 + By) + C

Things To Note:
    + Once we start moving to multivariate quadratic regression, or even multivariate polynomial regression, many 
    derivatives must be calculated. Finding a large number of derivatives will become tedious, and in many cases 
    impossible. A ML framework must be implemented in such cases. 

Moving On:
    + Implement Multivariate Quadratic Regression
    + Implement a ML framework!

"""


# ----------------------------------------------------------------------------------------------------------------------


def gradient_descent(x1, x2, y, batch):
    # Time Functions
    start_time = time.time()

    # Start With Some Value For Ax1, Bx1, Ax2, Bx2, and Cx
    ax1_curr = random.randint(0, 100)
    bx1_curr = random.randint(0, 100)
    ax2_curr = random.randint(0, 100)
    bx2_curr = random.randint(0, 100)
    cx_curr = random.randint(0, 100)

    # Number Of Iterations
    iterations = 10000

    # Define Learning Rate
    learning_rate = 0.05

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
        y_predicted = ((ax1_curr * (x1_batch**2)) + (bx1_curr * x1_batch)) +\
                      ((ax2_curr * (x2_batch**2)) + (bx2_curr * x2_batch)) + cx_curr

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


# Instantiate Number Of Points, And Batch Size
points = 10000
batch = 20

# Set Values Of Real Parameters
a1_actual = 5
b1_actual = 2
a2_actual = 5
b2_actual = 2
c_actual = 1

# Set X1, X2 Values
x1 = 2 * (np.random.rand(1, points))
x2 = 2 * (np.random.rand(1, points))

# Y Follows The Equation F(x, y) = (Ax^2 + Bx) + (Ay^2 + By) + C
y = ((a1_actual * (x1**2)) + (b1_actual * x1)) + ((a2_actual * (x2**2)) + (b2_actual * x2)) + c_actual

# Turn To Simple Lists
x1_basic = [num for num in x1[0]]
x2_basic = [num for num in x2[0]]
y_basic = [num for num in y[0]]

x1 = np.array(x1_basic)
x2 = np.array(x2_basic)
y = np.array(y_basic)

# Instantiate, And Collect Data
gradient_descent(x1, x2, y, batch)

# Instantiate 3D Plot
fig = plt.figure(figsize=plt.figaspect(0.50))

# Plot Of Data - 3D | SubPlot 1|
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(x1_basic, x2_basic, y_basic, marker="+", s=0.5)
plt.show()
