

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score



columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight",
           "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]


data = pd.read_csv("abalone.data", names=columns)



data = pd.get_dummies(data, columns=["Sex"])


X = data.drop("Rings", axis=1).values
y = data["Rings"].values.reshape(-1, 1)


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


np.random.seed(42)

input_size = X_train.shape[1]
hidden_size = 16
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))



learning_rate = 0.001
epochs = 500

losses = []

for epoch in range(epochs):

   
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    output = z2 

    
    loss = mse(y_train, output)
    losses.append(loss)


    d_output = -2 * (y_train - output) / y_train.shape[0]

    dW2 = np.dot(a1.T, d_output)
    db2 = np.sum(d_output, axis=0, keepdims=True)

    da1 = np.dot(d_output, W2.T)
    dz1 = da1 * relu_derivative(z1)

    dW1 = np.dot(X_train.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)


    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Training Loss: {loss:.4f}")



z1_test = np.dot(X_test, W1) + b1
a1_test = relu(z1_test)
z2_test = np.dot(a1_test, W2) + b2
test_output = z2_test

test_mse = mse(y_test, test_output)
r2 = r2_score(y_test, test_output)


print(f"Final Training Loss: {losses[-1]:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"R2 Score: {r2:.4f}")



plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss (MSE)")
plt.title("Training Loss Curve")
plt.show()


predicted_age = test_output + 1.5
actual_age = y_test + 1.5

print("\nSample Predictions (First 5):")
for i in range(5):
    print(f"Predicted Age: {predicted_age[i][0]:.2f}, Actual Age: {actual_age[i][0]:.2f}")
