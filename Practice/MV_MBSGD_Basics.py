# Name:                                            Renacin Matadeen
# Student Number:                                        N/A
# Date:                                              09/29/2018
# Course:                                                N/A
# Title                                      Multivariate Linear Regression
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

In many analyses, multiple variables make up the basis of the experiment. As a result if would be beneficial to look at
an implementation of Multivariate-Linear Regression. Note that this experiment will use Mini-Batch Stochastic Gradient
Descent

Next Steps, Look Towards implementations In TensorFlow, or other ML frameworks!

"""


# ----------------------------------------------------------------------------------------------------------------------


def gradient_descent(x1, x2, y):
    # Time Functions
    start_time = time.time()

    # Start With Some Value For m & b
    m1_curr = random.randint(0, 100)
    m2_curr = random.randint(0, 100)
    b_curr = random.randint(0, 100)

    # Number Of Iterations
    iterations = 100

    # Define Learning Rate
    learning_rate = 0.05

    # Print Heading
    print("{:-^55}".format("Gradient Descent"))

    # Gather Data
    cost_values = []
    m1_values = []
    m2_values = []
    b_values = []

    # Batch Size
    n = 50

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


# Populate X, Y with numbers generated with a set function, however gaussian noise has been added

# Instantiate Number Of Points, And Parameters 1, 2, and the Bias 0
points = 2500
m1_actual = 2
m2_actual = 3
bias_value = 5

x1 = 2 * (np.random.rand(1, points))
x2 = 4 * (np.random.rand(1, points))

y = (bias_value + (m1_actual * x1) + (m2_actual * x2)) + (np.random.rand(1, points))

# Turn To Simple Lists
x1_basic = [num for num in x1[0]]
x2_basic = [num for num in x2[0]]
y_basic = [num for num in y[0]]

x1 = np.array(x1_basic)
x2 = np.array(x2_basic)
y = np.array(y_basic)

# Instantiate, And Collect Data
m1, m2, b, cost_values, m1_values, m2_values, b_values = gradient_descent(x1, x2, y)

print("\n{:-^55}".format("Plots"))
print("\n{:^55}".format("*See Plots*"))

# Instantiate 3D Plot
fig = plt.figure(figsize=plt.figaspect(0.50))

# Plot Of Data - 3D | SubPlot 1|
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(x1_basic, x2_basic, y_basic)
plt.title('Raw Data Visualized')
ax.set_xlabel('X1 Values')
ax.set_ylabel('X2 Values')
ax.set_zlabel('Y Values')

# Plot Of M1, M2, and B Values - 3D | SubPlot 2|
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax1.scatter(m1_values, m2_values, b_values)
plt.title('Parameter Iteration')
ax1.set_xlabel('M1')
ax1.set_ylabel('M2')
ax1.set_zlabel('Bias')
plt.show()

# Plot Of Cost Values, As A Line
plt.xlabel('Iteration')
plt.ylabel('Iteration Cost')
plt.title('Cost Values Over Iterations')
plt.grid(True)
iteration = [x for x in range(len(cost_values))]
plt.plot(iteration, cost_values)
plt.show()

# Misc
print("\n{:-^50}".format("🔥🔥🔥🔥🔥"))