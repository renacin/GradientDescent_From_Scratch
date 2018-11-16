# Name:                                            Renacin Matadeen
# Student Number:                                        N/A
# Date:                                              09/29/2018
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

Things To Note:
    + Once we start moving to multivariate quadratic regression, or even multivariate polynomial regression, many 
    derivatives must be calculated. Finding a large number of derivatives will become tedious, and in many cases 
    impossible. A ML framework must be implemented in such cases. 

Moving On:
    + Implement Multivariate Quadratic Regression
    + Implement a ML framework!
    
"""
# ----------------------------------------------------------------------------------------------------------------------


def gradient_descent(x, y, batch):
    # Time Functions
    start_time = time.time()

    # Start With Some Value For m & b
    ax_curr = 0
    bx_curr = 0
    cx_curr = 0

    # Number Of Iterations
    iterations = 100

    # Define Learning Rate
    learning_rate = 0.05

    # Print Heading
    print("{:-^55}".format("Gradient Descent"))

    # Gather Data
    cost_values = []
    ax_values = []
    bx_values = []
    cx_values = []

    # Batch Size
    n = batch

    # Iteration
    for i in range(iterations):

        # Get Random Points In Dataset, Takes Batch Size
        x_batch = []
        y_batch = []

        # Can I Use String Formatting?
        for num in range(0, n):
            r_int = int(random.randint(1, x.size))
            x_batch.append(x.item(r_int - 1))
            y_batch.append(y.item(r_int - 1))

        # Turn Lists To Np.Arrays
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        # Find The Y Predicted Given The Equation AX^2 + BX + C, and Curr Values
        y_predicted = (ax_curr * (x_batch**2)) + (bx_curr * x_batch) + cx_curr

        # Find The Cost Of The Current Model, Using List Comprehension, Works Because Of Np.Array, Not Python List
        # Y & Y_Predicted Are Lists, Np.Array Allows For Matrix Subtraction
        cost = (1/n) * sum([val**2 for val in (y_batch - y_predicted)])

        # Find The Derivative Of Parameter A, B, And C, Given Equation AX^2 + BX + C
        ax_md = -(2/n) * sum((x_batch ** 2) * (y_batch - y_predicted))
        bx_md = -(2/n) * sum(x_batch * (y_batch - y_predicted))
        cx_md = -(2/n) * sum(y_batch - y_predicted)

        # Adjust m & b values. Learning Rate Is Subtracted Because Descent In Gradient Descent
        ax_curr = ax_curr - learning_rate * ax_md
        bx_curr = bx_curr - learning_rate * bx_md
        cx_curr = cx_curr - learning_rate * cx_md

        # Print Values
        print("Iteration {0} | Cost: {1:.3f} | A: {2:.3f} | B: {3:.3f} | C: {4:.3f}"
              .format(i + 1, cost, ax_curr, bx_curr, cx_curr))

        cost_values.append(cost)
        ax_values.append(ax_curr)
        bx_values.append(bx_curr)
        cx_values.append(cx_curr)

    # End Timer
    total_time = time.time() - start_time
    print("\n{:-^55}".format("Estimated Time"))
    print("Seconds: {0:.2f}".format(total_time))
    return ax_values, bx_values, cx_values, cost_values

# ----------------------------------------------------------------------------------------------------------------------


# Instantiate Number Of Points, A, B, C, and Batch Size
points = 10000
a_actual = 2
b_actual = 5
c_actual = 10
batch = 5

# Populate X, Y with numbers generated with a set function, however gaussian noise has been added
x = 2 * (np.random.rand(1, points))
y = ((a_actual * (x**2)) + (b_actual * x) + c_actual) + (np.random.rand(1, points))

# Turn To Simple Lists
x_basic = [num for num in x[0]]
y_basic = [num for num in y[0]]
x = np.array(x_basic)
y = np.array(y_basic)

# Instantiate, And Collect Data
ax_values, bx_values, cx_values, cost = gradient_descent(x, y, batch)

# Plot Data
print("\n{:-^55}".format("Plots"))
print("\n{:^55}".format("*See Plots*"))

# Plot Of Data
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Plot Of Data')
plt.grid(True)
plt.scatter(x_basic, y_basic, marker="o", s=1)
plt.show()

# Instantiate 3D Plot
fig = plt.figure(figsize=plt.figaspect(0.50))

# Plot Of Constants - 3D | SubPlot 1|
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(ax_values, bx_values, cx_values)
plt.title('Iteration Of Constants')
ax.set_xlabel('A Values')
ax.set_ylabel('B Values')
ax.set_zlabel('C Values')
plt.show()

# Plot Of Cost Values, As A Line
skip = 0
plt.xlabel('Iteration')
plt.ylabel('Iteration Cost')
plt.title('Cost Values Over Iterations')
plt.grid(True)
iteration = [x for x in range(len(cost))]
plt.plot(iteration[skip:], cost[skip:])
plt.show()


