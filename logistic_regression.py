import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0)

# Initialize and train the model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression')
plt.grid(True)
plt.show()
