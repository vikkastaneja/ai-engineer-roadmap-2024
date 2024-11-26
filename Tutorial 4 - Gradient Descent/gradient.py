import numpy as np
import matplotlib.pyplot as plt
import math

# In this exercise, we will develop gradient descent algorithm that may be used to by linear regression model to calculate the line of best fit
# We will use single variable as an example
# https://www.youtube.com/watch?v=Aym1Gx2wVK4&list=PLPbgcxheSpE0aBsefANDYe2X_-tyJbBMr&index=4
# y=mx+b --> goal is to find m and b using gradient descent method
# we will use partial derivatives to find out m and b using mean square values for each step to getting close to m and b separately.
# So for each step, m = m - step size and b = b - step size
# step size = learning rate * delta where delta is partial derivate of m or b
# m = -(2/n)* sum of product of x and diff y
# b = -(2/n)* sum of diff y
# cost = mse = (1/n)sigma((y-y_prediction)^2)
# Once the cost is almost zero, we find one potential m and b
def gradient_descent_algorithm(x, y):
    m_current = b_current = 0

    n = len(x)
    learning_rate = 0.0002
    iteration = 500000
    cost = 10
    cost_previous = 100
    for i in range(iteration):
        y_predicted = m_current * x + b_current
        # plt.plot(x, y_predicted, color='b', label=i)

        # Calculate the mean square error, which is the cost
        # mse = (1/n)sigma((y-y_predicted)**2)
        cost = (1/n) * sum([val ** 2 for val in (y - y_predicted)])
        m_derivative = -(2/n)* sum (x * (y-y_predicted))
        b_derivative = -(2/n)* sum (y-y_predicted)

        m_current = m_current - learning_rate * m_derivative
        b_current = b_current - learning_rate * b_derivative
        # print(f'm: {m_current}, b: {b_current}, cost: {cost}, iteration: {i}')
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        i += 1

    # plt.plot(x, y_predicted, c='blue')
    plt.scatter(x, y, c='red')
    plt.scatter(x, y_predicted, c='green')
    # plt.show()
    return (m_current, b_current, i-1)

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

x = np.array([92,56,88,70,80,49,65,35,66,67])
y = np.array([98,68,81,80,83,52,66,30,68,73])
coeff_intercept = gradient_descent_algorithm(x, y)

# Check what linear model gives for the same x and y
from sklearn import linear_model
import pandas as pd
import math
model = linear_model.LinearRegression()

arr = np.array([x,y])
df = pd.DataFrame(arr.T, columns=["X","Y"])
model.fit(df.drop('Y', axis=1), df.drop('X', axis=1))
np.testing.assert_allclose(round(model.coef_[0][0], 2), round(coeff_intercept[0], 2))
np.testing.assert_allclose(round(model.intercept_[0], 2), round(coeff_intercept[1], 2))

# Create a range of x values for the line
x_line = np.linspace(x.min(), x.max(), 100)

# Calculate the corresponding y values for the line
y_line = round(model.coef_[0][0], 2) * x_line + round(model.intercept_[0], 2)

# Plot the data points and the regression line
plt.scatter(x, y)
plt.plot(x_line, y_line, color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
# plt.show()
