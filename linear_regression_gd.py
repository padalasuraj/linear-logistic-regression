# linear_regression_gd.py

import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * X.T.dot(errors)
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
    y = np.array([2, 3, 5, 7, 11])
    theta = np.zeros(X.shape[1])

    # Parameters for Gradient Descent
    alpha = 0.01  # Learning rate
    num_iters = 1000

    # Running Gradient Descent
    theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)

    print(f'Theta found by gradient descent: {theta}')
    print(f'Final cost: {cost_history[-1]}')
