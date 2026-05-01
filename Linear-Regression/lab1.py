import numpy as np

# ============================================================
# CS229 Lecture 2 Lab: Multiple Linear Regression
# Implementations:
# 1. Hypothesis Model
# 2. Cost Function
# 3. Gradient Descent
# 4. Normal Equation
# Dataset: Multi-feature (Size, Bedrooms, Age)
# ============================================================


# ============================================================
# STEP 1: Create Dataset 
# Features:
# X1 = Size in sqft
# X2 = Bedrooms
# X3 = Age of house in years
# Target:
# y  = Price in $1000s
# ============================================================

X = np.array([
    [2100, 3, 20],
    [1600, 2, 15],
    [2400, 4, 30],
    [1400, 2, 10],
    [3000, 4, 8],
    [1985, 3, 12],
    [1534, 2, 25],
    [1427, 2, 18]
])

y = np.array([400, 330, 369, 342, 540, 410, 320, 350])

m = X.shape[0]   # number of training examples
n = X.shape[1]   # number of features

print("Training examples (m):", m)
print("Features (n):", n)


# ============================================================
# STEP 2: Feature Normalization (VERY IMPORTANT for GD)
# Normalize each feature:
# X_norm = (X - mean) / std
# ============================================================

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X_norm = (X - X_mean) / X_std

print("\nFeature Means:", X_mean)
print("Feature Std Dev:", X_std)


# ============================================================
# STEP 3: Add Bias Term (x0 = 1)
# This makes theta include theta0 (intercept)
# X_bias shape becomes (m, n+1)
# ============================================================

X_bias = np.c_[np.ones(m), X_norm]

print("\nX_bias shape:", X_bias.shape)


# ============================================================
# STEP 4: Hypothesis Function (CS229)
# h_theta(x) = X * theta
# ============================================================

def hypothesis(X, theta):
    return X @ theta


# ============================================================
# STEP 5: Cost Function (Squared Error Cost)
# J(theta) = (1/2m) * sum( (h(x) - y)^2 )
# ============================================================

def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    error = predictions - y
    cost = (1/(2*m)) * np.sum(error**2)
    return cost


# ============================================================
# STEP 6: Gradient Descent (Vectorized Implementation)
# Update rule:
# theta = theta - alpha * (1/m) * X^T * (X*theta - y)
# ============================================================

def gradient_descent(X, y, theta, alpha=0.1, iterations=1000):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = hypothesis(X, theta)
        error = predictions - y

        # gradient calculation
        gradient = (1/m) * (X.T @ error)

        # update theta
        theta = theta - alpha * gradient

        # store cost for tracking
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Print cost every 100 iterations (CS229 debugging)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")

    return theta, cost_history


# ============================================================
# STEP 7: Train Model Using Gradient Descent
# Initialize theta as zeros
# ============================================================

theta_init = np.zeros(n + 1)

theta_gd, cost_history = gradient_descent(
    X_bias, y, theta_init,
    alpha=0.1,
    iterations=1000
)

print("\nTheta from Gradient Descent:")
print(theta_gd)


# ============================================================
# STEP 8: Normal Equation (CS229 Closed Form)
# theta = pinv(X) * y
# (Pseudo-inverse avoids matrix inverse errors)
# ============================================================

def normal_equation(X, y):
    theta = np.linalg.pinv(X) @ y
    return theta


theta_ne = normal_equation(X_bias, y)

print("\nTheta from Normal Equation:")
print(theta_ne)


# ============================================================
# STEP 9: Compare Cost (GD vs Normal Equation)
# ============================================================

cost_gd = compute_cost(X_bias, y, theta_gd)
cost_ne = compute_cost(X_bias, y, theta_ne)

print("\nFinal Cost using Gradient Descent:", cost_gd)
print("Final Cost using Normal Equation:", cost_ne)


# ============================================================
# STEP 10: Prediction Function
# Must normalize input using training mean/std
# ============================================================

def predict_price(size, bedrooms, age, theta):
    x = np.array([size, bedrooms, age])

    # normalize input features
    x_norm = (x - X_mean) / X_std

    # add bias term
    x_input = np.insert(x_norm, 0, 1)

    # prediction
    return x_input @ theta


# ============================================================
# STEP 11: Test Predictions
# ============================================================

test_house_1 = predict_price(2500, 4, 5, theta_gd)
test_house_2 = predict_price(2500, 4, 5, theta_ne)

print("\nPrediction for house (2500 sqft, 4 bedrooms, 5 years old):")
print("Gradient Descent Prediction:", test_house_1)
print("Normal Equation Prediction:", test_house_2)

test_house_3 = predict_price(3000, 5, 2, theta_ne)
print("\nPrediction for house (3000 sqft, 5 bedrooms, 2 years old):")
print("Normal Equation Prediction:", test_house_3)
