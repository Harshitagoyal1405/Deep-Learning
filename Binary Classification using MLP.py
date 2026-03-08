import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("diabetes.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1,1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

np.random.seed(42)

input_size = X_train.shape[1]
h1 = 64
h2 = 32
output_size = 1

W1 = np.random.randn(input_size, h1) * np.sqrt(2/input_size)
b1 = np.zeros((1, h1))

W2 = np.random.randn(h1, h2) * np.sqrt(2/h1)
b2 = np.zeros((1, h2))

W3 = np.random.randn(h2, output_size) * np.sqrt(2/h2)
b3 = np.zeros((1, output_size))

learning_rate = 0.005
epochs = 4000
losses = []

for epoch in range(epochs):

    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    output = sigmoid(z3)

    loss = -np.mean(y_train*np.log(output+1e-8) +
                    (1-y_train)*np.log(1-output+1e-8))
    losses.append(loss)

    d3 = output - y_train

    dW3 = np.dot(a2.T, d3) / X_train.shape[0]
    db3 = np.sum(d3, axis=0, keepdims=True) / X_train.shape[0]

    d2 = np.dot(d3, W3.T) * relu_derivative(z2)
    dW2 = np.dot(a1.T, d2) / X_train.shape[0]
    db2 = np.sum(d2, axis=0, keepdims=True) / X_train.shape[0]

    d1 = np.dot(d2, W2.T) * relu_derivative(z1)
    dW1 = np.dot(X_train.T, d1) / X_train.shape[0]
    db1 = np.sum(d1, axis=0, keepdims=True) / X_train.shape[0]

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 800 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

z1_test = np.dot(X_test, W1) + b1
a1_test = relu(z1_test)

z2_test = np.dot(a1_test, W2) + b2
a2_test = relu(z2_test)

z3_test = np.dot(a2_test, W3) + b3
test_output = sigmoid(z3_test)

y_pred = (test_output > 0.42).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()