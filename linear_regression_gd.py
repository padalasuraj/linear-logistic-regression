import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.m = len(y)
        self.X = X
        self.y = y
        self.theta = np.zeros((X.shape[1] + 1, 1))

        X_b = np.c_[np.ones((self.m, 1)), X]
        for _ in range(self.n_iterations):
            gradients = 2/self.m * X_b.T.dot(X_b.dot(self.theta) - y.reshape(-1, 1))
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Initialize and train the model
model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[6], [7], [8]]))

print(predictions)  # Output: array([12., 14., 16.])

# Plotting
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
