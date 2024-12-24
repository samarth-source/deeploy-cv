import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Input the number of features and data points
num_features = int(input("Enter the number of features: "))
num_points = int(input("Enter the number of data points: "))

# Input the data
print(f"Enter the feature values for {num_points} data points, each with {num_features} features:")
X = []
for i in range(num_points):
    row = list(map(float, input(f"Data point {i+1}: ").split()))
    X.append(row)

print(f"Enter the target values for the {num_points} data points:")
y = []
for i in range(num_points):
    y.append(float(input(f"Target value for data point {i+1}: ")))

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Ask for ridge regularization parameter
alpha = float(input("Enter the regularization strength (alpha): "))

# Fit the Ridge regression model
model = Ridge(alpha=alpha)
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Print results
print("\nRidge Regression Results:")
print("Coefficients (weights):", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y, y_pred))
