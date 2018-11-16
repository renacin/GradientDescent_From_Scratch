# Name:                                            Renacin Matadeen
# Student Number:                                        N/A
# Date:                                              10/03/2018
# Course:                                                N/A
# Title                               Understanding Gradient Descent With Momentum
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

Another iteration of the Gradient Descent Optimization function is MSGD with Momentum. To elaborate, in the original
rendition of GD, smaller and smaller steps are taken as the optimizer gets closer to the optimal gradient. This may be
beneficial in the case of accuracy, and not wanting to overshoot, however this is extremely costly and inefficient in
regards to some of the vast ravines that must be travelled in some cases. This experiment will attempt to implement a 
Momentum based MSGD.

Figure 1 - Table Of Optimizer Performance

               |-------------------------------------------------------------------------|
               | Optimization Function | Time (Seconds) | Final Cost | Final M | Final B |
               |-------------------------------------------------------------------------|
               | Batch Gradient Descent|      79.82     |    0.08    |   1.99  |  5.50   |
               |-------------------------------------------------------------------------|
               |     Stochastic GD     |       0.02     |    0.10    |   2.12  |  5.50   |
               |-------------------------------------------------------------------------|
               |     Mini-Batch SGD    |       0.34     |    0.07    |   1.99  |  4.49   |
               |-------------------------------------------------------------------------|

"""


# ----------------------------------------------------------------------------------------------------------------------


def gradient_descent(x, y, b_friction, batch, learning_rate):
    # Time Functions
    start_time = time.time()

    # Start With Some Value For m, b, and momentum
    m_curr = 0
    b_curr = 0
    momentum_m = 0
    momentum_b = 0

    # Number Of Iterations
    iterations = 1000



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

        # Find The Cost Of The Current Model
        cost = (1 / n) * sum([val ** 2 for val in (y_batch - y_predicted)])

        # Find The m & b derivative
        md = -(2 / n) * sum(x_batch * (y_batch - y_predicted))
        bd = -(2 / n) * sum(y_batch - y_predicted)

        # Adjust m & b values. Learning Rate Is Subtracted Because Descent In Gradient Descent, Implement Momentum!
        momentum_m = (b_friction * momentum_m) - (learning_rate * md)
        m_curr += momentum_m

        momentum_b = (b_friction * momentum_b) - (learning_rate * bd)
        b_curr += momentum_b

        # Print Values
        print("Iteration {0} | Cost: {1:.5f} | M: {2:.5f} | B: {3:.5f}".format(i + 1, cost, m_curr, b_curr))

        cost_values.append(cost)
        m_values.append(m_curr)
        b_values.append(b_curr)

    # End Timer
    total_time = time.time() - start_time
    print("\n{:-^55}".format("Estimated Time"))
    print("Seconds: {0:.2f}".format(total_time))
    return cost_values, m_values, b_values


# ----------------------------------------------------------------------------------------------------------------------


# Instantiate Number Of Points, Batch Size, and Friction_Coefficient
points = 100000
batch = 10
friction = 0.99
learning_rate = 0.01

# Value Of M, & B actual
m_actual = 2
b_actual = 5

# Instantiate X, and Y values
x = 2 * (np.random.rand(1, points))
y = (b_actual + (m_actual * x)) + (np.random.rand(1, points))

# Turn To Simple Lists
x_basic = [num for num in x[0]]
y_basic = [num for num in y[0]]
x = np.array(x_basic)
y = np.array(y_basic)

# Instantiate, And Collect Data
cost_values, m_values, b_values = gradient_descent(x, y, friction, batch, learning_rate)

# Plot Of Cost Values, As A Line
plt.xlabel('Iteration')
plt.ylabel('Iteration Cost')
plt.title('Cost Values Over Iterations')
plt.grid(True)
iteration = [x for x in range(len(cost_values))]
plt.plot(iteration, cost_values)
plt.show()




