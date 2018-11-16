# Name:                                            Renacin Matadeen
# Student Number:                                        N/A
# Date:                                              09/21/2018
# Course:                                                N/A
# Title                                      Understanding Gradient Descent
#
#
#
#
#
# ----------------------------------------------------------------------------------------------------------------------

import time
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
"""

In order to understand machine learning, its inner workings, and how it can be used an individual must first understand 
the basics of the field. One of the most important concepts in machine learning is the ability to learn from data, and 
make actions based on them. In its simplest form gradient descent provides computers with the ability to iteratively
find the lowest cost for a specific function. This is important as machines need a method of determining which answer is
the most appropriate given a task. 

As a note:
    Gradient Descent Implements:
        + Calculus
            - Derivatives
            - Partial Derivatives
            
        + Programmatic Math
            - Effectively Implementing Equations
            
Things To Remember:
    Mean Squared Error
        + Gradient descent looks to minimize error, or cost of the overall prediction to ensure optimal accuracy. 
        The distance between a sample point and the line of best fit is considered the error, note that the error is 
        squared to account for negative values as well as return a observation of magnitude. 
        
        The Formula For MSE Is:

                1/N * Σ(y_i - y_predicted)^2
                
                Where:
                    y_predicted = (mx_i + b)
                    
        When you're trying to minimize cost, your answer depends on two other variables (m, and b)
        The theoretical space becomes 3-Dimensional. 
        
    Always look for the global minimum, never the local minimum!
    
    As you get closer to the minimum, make smaller steps. This is important as you don't want to overshoot your
    descent! Find the slope to find the direction, and as the slope decreases make smaller movements. To do this you
    must find partial derivatives!
    
        Slope Iteration:
            b2 = b1 - Learning Rate * Partial Derivative (Slope)
    
    In this case find the partial derivatives of MSE!
    
Taking A Step Back
    This method of gradient descent is quite accurate. However it uses all data points to calculate the line of best
    fit. In the case of 10 or so data points this isn't bad. But when scaled up to 1'000'000 if not billions of points
    this can take some time to complete. Additionally learning rates, as well as iterations also add another level of 
    time complexity. 
    
    Looking to solve this issue you can use Stochastic Gradient Descent (SGD), if not Mini-Batch SGD to solve such an
    issue. 
    
Next Steps, Look Towards implementations In TensorFlow, or other ML frameworks!

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

    # Length Of Data Points
    n = len(x)

    # Define Learning Rate
    learning_rate = 0.05

    # Print Heading
    print("{:-^55}".format("Gradient Descent"))

    # Gather Data
    cost_values = []
    m_values = []
    b_values = []

    # Iteration
    for i in range(iterations):

        # Find The Y Predicted Given The m & b
        y_predicted = m_curr * x + b_curr

        # Find The Cost Of The Current Model, Using List Comprehension, Works Because Of Np.Array, Not Python List
        # Y & Y_Predicted Are Lists, Np.Array Allows For Matrix Subtraction
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])

        # Find The m & b derivative
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)

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
points = 1000
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

# # Prediction: X Values To Map
# x1 = 1
# x2 = 5
# x3 = 200
#
# # Find Values
# y1 = (m * x1) + b
# y2 = (m * x2) + b
# y3 = (m * x3) + b
#
# # Print Additional Info For Understanding
# print("\n{:-^55}".format("Predictions"))
# print("If: {0} = {1:.2f} | {2} = {3:.2f} | {4} = {5:.2f}".format(x1, y1, x2, y2, x3, y3))
#
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
