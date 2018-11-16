# Name:                                            Renacin Matadeen
# Student Number:                                        N/A
# Date:                                              09/27/2018
# Course:                                                N/A
# Title                                 Understanding Stochastic Gradient Descent
#
#
#
#
#
# ----------------------------------------------------------------------------------------------------------------------

import time
import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
"""

As a reminder, machine learning depends on the ability to effectively learn from mistakes, and make optimizations. In 
the first test we looked at Batch Gradient Descent as an optimization methodology. From subsequent tests, we were able
to smoothly appropriate both a slope, and an intercept that minimized the Mean Square Error function. As a note however,
a number of concerns did arise. Most notably was that Batch Gradient Descent is not a scalable optimization technique. 
In other words, since the technique requires the error to be calculated from all data points, in a scenario where there
is 100'000+ data points, a great deal of computing power would be wasted on something simple. To combat such an issue an
individual can implement Stochastic Gradient Descent, or the process of iterating one's gradient for one random point.
In the following program stochastic gradient descent will be implemented. 


Results Of Batch Gradient Descent Vs Stochastic Gradient Descent:

               |-------------------------------------------------------------------------|
               |Optimization Function  | Time (Seconds) | Final Cost | Final M | Final B |
               |-------------------------------------------------------------------------|
               |Batch Gradient Descent |      79.82     |    0.08    |   1.99  |  5.50   |
               |-------------------------------------------------------------------------|
               |    Stochastic GD      |       0.02     |    0.10    |   2.12  |  5.50   |
               |-------------------------------------------------------------------------|

From the results drawn it seems that Stochastic Gradient Descent is far superior in terms of efficiency, as well as 
sheer speed. As a note however, a number of criticisms nonetheless arise, most notably from the varying results as
iterations pass. It would be of great benefit to explore a Mini-Batch Stochastic Gradient Descent.

Notes:
    How do I measure a model's accuracy?

"""
# ----------------------------------------------------------------------------------------------------------------------


def gradient_descent(x, y):
    # Time Functions
    start_time = time.time()

    # Start With Some Value For m & b
    m_curr = 0
    b_curr = 0

    # Number Of Iterations
    iterations = 1000

    # Define Learning Rate
    learning_rate = 0.05

    # Print Heading
    print("{:-^55}".format("Gradient Descent"))

    # Gather Data
    cost_values = []
    m_values = []
    b_values = []

    # Batch Size
    n = 1

    # Iteration
    for i in range(iterations):

        # Get Random Point In Dataset
        r_int = int(random.randint(1, x.size))
        x_value = x.item(r_int - 1)
        y_value = y.item(r_int - 1)

        # Find The Y Predicted Given The m & b
        y_predicted = m_curr * x_value + b_curr

        # Find The Cost Of The Current Model, Using List Comprehension, Works Because Of Np.Array, Not Python List
        # Y & Y_Predicted Are Lists, Np.Array Allows For Matrix Subtraction
        cost = (1/n) * (y_value - y_predicted)**2

        # Find The m & b derivative, The Derivative Doesn't Change Much. Just Remove The Sum, And N = The Batch Size
        md = -(2/n) * (x_value * (y_value - y_predicted))
        bd = -(2/n) * (y_value - y_predicted)

        # Adjust m & b values. Learning Rate Is Subtracted Because Descent In Gradient Descent
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        # Print Values
        print("Iteration {0} | Cost: {1:.5f} | M: {2:.5f} | B: {3:.5f}".format(i + 1, cost, m_curr, b_curr))

        cost_values.append(cost)
        m_values.append(m_curr)
        b_values.append(b_curr)

    # End Timer
    total_time = time.time() - start_time
    print("\n{:-^55}".format("Estimated Time"))
    print("Seconds: {0:.2f}".format(total_time))
    return m_curr, b_curr, cost_values, m_values, b_values

# ----------------------------------------------------------------------------------------------------------------------


# Populate X, Y with numbers generated with a set function, however gaussian noise has been added

# Instantiate Number Of Points, M, and B
points = 100000
m_actual = 2
b_actual = 5

x = 2 * (np.random.rand(1, points))
y = (b_actual + (m_actual * x)) + (np.random.rand(1, points))

# Turn To Simple Lists
x_basic = [num for num in x[0]]
y_basic = [num for num in y[0]]

x = np.array(x_basic)
y = np.array(y_basic)

# Instantiate, And Collect Data
m, b, cost_values, m_values, b_values = gradient_descent(x, y)

# Prediction: X Values To Map
x1 = 1
x2 = 5
x3 = 200

# Find Values
y1 = (m * x1) + b
y2 = (m * x2) + b
y3 = (m * x3) + b

# Print Additional Info For Understanding
print("\n{:-^55}".format("Predictions"))
print("If: {0} = {1:.2f} | {2} = {3:.2f} | {4} = {5:.2f}".format(x1, y1, x2, y2, x3, y3))

print("\n{:-^55}".format("Plots"))
print("\n{:^55}".format("*See Plots*"))


# Plot Of Data
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Plot Of Data')
plt.grid(True)
plt.scatter(x_basic, y_basic, marker="o", s=1)
plt.show()

# Plot Of Cost Values, As A Line
plt.xlabel('Iteration')
plt.ylabel('Iteration Cost')
plt.title('Cost Values Over Iterations')
plt.grid(True)
iteration = [x for x in range(len(cost_values))]
plt.plot(iteration, cost_values)
plt.show()

# Plot Of M By B Values, As A Line
plt.xlabel('B Values')
plt.ylabel('M Values')
plt.title('Gradient Descent')
plt.grid(True)
plt.plot(b_values, m_values)
plt.show()

# Misc
print("\n{:-^50}".format("🔥🔥🔥🔥🔥"))