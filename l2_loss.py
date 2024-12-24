import numpy as np

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

# Add a column of ones to X for the intercept (bias term)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Ask for the regularization strength
alpha = float(input("Enter the regularization strength (alpha): "))

# Initialize weights (including intercept)
weights = np.random.randn(X_b.shape[1])

# Learning parameters
learning_rate = 0.01
iterations = 1000

# L2 Regularized Loss (Cost) function
def l2_regularized_loss(X, y, weights, alpha):
    m = len(y)
    predictions = X.dot(weights)
    error = predictions - y
    # Regularized loss: (MSE + L2 penalty term)
    mse_loss = (1/(2 * m)) * np.sum(error ** 2)
    l2_penalty = (alpha / 2) * np.sum(weights[1:] ** 2)  # Exclude intercept from L2 penalty
    return mse_loss + l2_penalty

# Gradient of the L2 regularized loss function
def gradient(X, y, weights, alpha):
    m = len(y)
    predictions = X.dot(weights)
    error = predictions - y
    gradient = (1/m) * X.T.dot(error)  # Gradient of MSE
    # Regularization term gradient (exclude intercept)
    gradient[1:] += alpha * weights[1:]
    return gradient

# Gradient Descent
for i in range(iterations):
    grad = gradient(X_b, y, weights, alpha)
    weights -= learning_rate * grad
    if i % 100 == 0:
        cost = l2_regularized_loss(X_b, y, weights, alpha)
        print(f"Iteration {i}: Cost = {cost}")

# Output final weights and cost
print("\nFinal weights:", weights)
final_cost = l2_regularized_loss(X_b, y, weights, alpha)
print("Final regularized cost:", final_cost)
