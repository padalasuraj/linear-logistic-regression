# 🧠 **Linear Regression Project**

## 📚 **Overview**

Welcome to the **Linear Regression Project**! This repository contains implementations of Linear Regression using both analytical methods and Gradient Descent. The project aims to provide an in-depth understanding of how linear regression works under the hood.

## 🏗️ **Project Structure**

```plaintext
📦 linear-regression-project
├── 📄 README.md                   # Project documentation
├── 📜 linear_regression.py         # Linear regression using normal equation
├── 📜 linear_regression_gd.py      # Linear regression using gradient descent
└── 📜 logical_regression.py        # Logistic regression implementation
🚀 Getting Started
Prerequisites
Make sure you have Python 3.x and the following libraries installed:pip install numpy matplotlib scikit-learn
Running the Scripts
Linear Regression with Normal Equation:

Run the following command to execute the script:python linear_regression.py
Linear Regression with Gradient Descent:

Run the following command to execute the script:python linear_regression_gd.py
🎨 Example Usage
Here’s a simple example of how to use the linear_regression_gd.py script:from linear_regression_gd import LinearRegressionGD
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Initialize the model
model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)

# Train the model
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[6], [7], [8]]))

print(predictions)  # Output: array([12., 14., 16.])
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sensitive Touch Panel</title>
    <style>
        .touch-panel {
            width: 300px;
            height: 200px;
            background-color: #2c3e50;
            margin: 50px auto;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ecf0f1;
            font-size: 24px;
            text-align: center;
            transition: background-color 0.3s ease;
        }

        .touch-panel:hover {
            background-color: #3498db;
            animation: blink 0.5s step-end infinite;
        }

        @keyframes blink {
            50% {
                opacity: 0;
            }
        }
    </style>
</head>
<body>
    <div class="touch-panel">
        Hover Over Me!
    </div>
</body>
</html>
📊 Visualization
This project includes visualization tools to help you better understand the regression model's fit:

Scatter Plot: Displays data points.
Regression Line: Shows the best-fit line overlay.
To visualize the results, ensure matplotlib is installed, then run the script. The plot will be displayed or saved in the plots/ directory (if configured).

📏 Performance Evaluation
Evaluate the model’s performance using the Mean Squared Error (MSE). The script will print the MSE value, helping you understand the model's accuracy.

⚙️ Customization
Learning Rate: Modify the learning rate for Gradient Descent by changing the learning_rate parameter in linear_regression_gd.py.
Number of Iterations: Adjust the number of iterations for Gradient Descent by changing the n_iterations parameter.
Visualization Settings: Customize the plot appearance by modifying matplotlib settings in the script.
```
