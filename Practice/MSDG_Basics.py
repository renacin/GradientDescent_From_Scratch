# Name:                                            Renacin Matadeen
# Student Number:                                        N/A
# Date:                                              09/28/2018
# Course:                                                N/A
# Title                             Understanding Mini-Batch Stochastic Gradient Descent
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
the first previous test, we looked at Stochastic Gradient Descent as a method of reducing the wasted computing power, 
and accurately modelling a dataset. From the results drawn, as seen in Figure 1, SGD can greatly reduce the number of 
calculations made, while also remaining accurate. However after looking into the results of the optimization function, 
a number of criticism arose, most notably was that as iterations got closer to the optimal solution, noise introduced
from the random selection of points forced the solution to circle, if not bounce around. Not only was this wasteful in
terms of computational power, but also inefficient in regards to the optimal solution. 


Figure 1. - Results Of Batch Gradient Descent Vs Stochastic Gradient Descent:

               |-------------------------------------------------------------------------|
               |Optimization Function  | Time (Seconds) | Final Cost | Final M | Final B |
               |-------------------------------------------------------------------------|
               |Batch Gradient Descent |      79.82     |    0.08    |   1.99  |  5.50   |
               |-------------------------------------------------------------------------|
               |    Stochastic GD      |       0.02     |    0.10    |   2.12  |  5.50   |
               |-------------------------------------------------------------------------|

In an attempt to solve such an issue, Mini-Batch Stochastic Gradient Descent, or MSGD, the process of using N number of 
points for each iteration, rather than all or just one, will be used in order to obtain more optimal solutions.

Figure 2. - Results Of Mini-Batch Stochastic Gradient Descent Vs Stochastic Gradient Descent:

               |-------------------------------------------------------------------------|
               | Optimization Function | Time (Seconds) | Final Cost | Final M | Final B |
               |-------------------------------------------------------------------------|
               | Batch Gradient Descent|      79.82     |    0.08    |   1.99  |  5.50   |
               |-------------------------------------------------------------------------|
               |     Stochastic GD     |       0.02     |    0.10    |   2.12  |  5.50   |
               |-------------------------------------------------------------------------|
               |     Mini-Batch SGD    |       0.34     |    0.07    |   1.99  |  4.49   |
               |-------------------------------------------------------------------------|
               
Looking at the results drawn from this experiment, it seems that MSGD is the best of both worlds, considering batch
gradient descent, and stochastic gradient descent. Put simply, the optimization function is able to combine the speed, 
and accuracy of both methodologies. Having said this however, there are still a number of techniques to explore. 

Things To Do:
    Look At
        + Momentum
        + Nesterov Accelerated Gradient (NAG)
        + Adagrad
        + Adadelta
        + ADAM

Notes:
    What is the optimal batch size for MSGD?
        + Joshua Benito, says 1 - 100, or in most cases 34. Depends on your sample size, and descriptive statistics. 
        
    How do I measure a model's accuracy?
        + Moving past just the cost function?
        + Training Set & Testing Set?
            - Research Needed
"""
# ----------------------------------------------------------------------------------------------------------------------


def gradient_descent(x, y, batch):
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

        # Find The Y Predicted Given The m & b, And X Values
        y_predicted = m_curr * x_batch + b_curr

        # Find The Cost Of The Current Model, Using List Comprehension, Works Because Of Np.Array, Not Python List
        # Y & Y_Predicted Are Lists, Np.Array Allows For Matrix Subtraction
        cost = (1/n) * sum([val**2 for val in (y_batch - y_predicted)])

        # Find The m & b derivative
        md = -(2/n) * sum(x_batch * (y_batch - y_predicted))
        bd = -(2/n) * sum(y_batch - y_predicted)

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

batch = 10

x = 2 * (np.random.rand(1, points))
y = (b_actual + (m_actual * x)) + (np.random.rand(1, points))

# Turn To Simple Lists
x_basic = [num for num in x[0]]
y_basic = [num for num in y[0]]

x = np.array(x_basic)
y = np.array(y_basic)

# Instantiate, And Collect Data
m, b, cost_values, m_values, b_values = gradient_descent(x, y, batch)
#
# # Prediction: X Values To Map
# x1 = 1
# x2 = 5
# x3 = 200
#
# # Find Values
# y1 = (m * x1) + b
# y2 = (m * x2) + b
# y3 = (m * x3) + b

# # Print Additional Info For Understanding
# print("\n{:-^55}".format("Predictions"))
# print("If: {0} = {1:.2f} | {2} = {3:.2f} | {4} = {5:.2f}".format(x1, y1, x2, y2, x3, y3))

# print("\n{:-^55}".format("Plots"))
# print("\n{:^55}".format("*See Plots*"))
#
#
# # Plot Of Data
# plt.xlabel('X Values')
# plt.ylabel('Y Values')
# plt.title('Plot Of Data')
# plt.grid(True)
# plt.scatter(x_basic, y_basic, marker="o", s=1)
# plt.show()

# Plot Of Cost Values, As A Line
plt.xlabel('Iteration')
plt.ylabel('Iteration Cost')
plt.title('Cost Values Over Iterations')
plt.grid(True)
iteration = [x for x in range(len(cost_values))]
plt.plot(iteration, cost_values)
plt.show()
#
# # Plot Of M By B Values, As A Line
# plt.xlabel('B Values')
# plt.ylabel('M Values')
# plt.title('Gradient Descent')
# plt.grid(True)
# plt.plot(b_values, m_values)
# plt.show()
#
# # Misc
# print("\n{:-^50}".format("🔥🔥🔥🔥🔥"))