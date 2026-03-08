# ============================================
# LAB 3: Multiple Linear Regression
# Using Single-Layer Perceptron
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv("multiple_linear_regression_dataset.csv")

print("Dataset Preview:")
print(data.head())

# ============================================
# 2. PREPROCESSING
# ============================================

# Features = all columns except last
# Target = last column
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Normalize features (important for gradient descent)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 3. INITIALIZE PERCEPTRON
# ============================================

np.random.seed(42)

input_size = X_train.shape[1]

W = np.random.randn(input_size, 1) * 0.1
b = np.zeros((1, 1))

learning_rate = 0.01
epochs = 500

losses = []

# ============================================
# 4. DEFINE LOSS FUNCTION
# ============================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ============================================
# 5. TRAINING (GRADIENT DESCENT)
# ============================================

for epoch in range(epochs):

    # Forward Pass (Perceptron Output)
    y_pred = np.dot(X_train, W) + b

    # Compute Loss
    loss = mse(y_train, y_pred)
    losses.append(loss)

    # Backward Pass (Gradients)
    dW = -2 * np.dot(X_train.T, (y_train - y_pred)) / X_train.shape[0]
    db = -2 * np.sum(y_train - y_pred) / X_train.shape[0]

    # Update Weights
    W -= learning_rate * dW
    b -= learning_rate * db

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ============================================
# 6. TESTING
# ============================================

y_test_pred = np.dot(X_test, W) + b

test_mse = mse(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\n==============================")
print("Training Complete")
print("==============================")
print(f"Final Training Loss: {losses[-1]:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# ============================================
# 7. PLOT LOSS CURVE
# ============================================

plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Curve")
plt.show()

# ============================================
# 8. SAMPLE PREDICTIONS (Safe Printing)
# ============================================

print("\nSample Predictions:")
for i in range(min(5, len(y_test_pred))):
    print(f"Predicted: {y_test_pred[i][0]:.2f}, Actual: {y_test[i][0]:.2f}")