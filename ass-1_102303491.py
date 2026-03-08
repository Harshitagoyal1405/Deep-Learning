import numpy as np

# ===============================
# SINGLE LAYER PERCEPTRON
# (AND and OR)
# ===============================

def step(x):
    return 1 if x >= 0 else 0

def train_perceptron(X, y, lr=0.1, epochs=20):
    n_features = X.shape[1]
    W = np.zeros(n_features)
    b = 0

    for _ in range(epochs):
        for i in range(len(X)):
            linear = np.dot(X[i], W) + b
            y_pred = step(linear)
            error = y[i] - y_pred
            W += lr * error * X[i]
            b += lr * error

    return W, b

def test_perceptron(X, W, b):
    for x in X:
        linear = np.dot(x, W) + b
        print(x, "->", step(linear))


X = np.array([[0,0],[0,1],[1,0],[1,1]])

# AND Gate
y_and = np.array([0,0,0,1])
W_and, b_and = train_perceptron(X, y_and)
print("AND Gate Output:")
test_perceptron(X, W_and, b_and)

# OR Gate
y_or = np.array([0,1,1,1])
W_or, b_or = train_perceptron(X, y_or)
print("\nOR Gate Output:")
test_perceptron(X, W_or, b_or)


# ===============================
# MULTI-LAYER PERCEPTRON
# (XOR)
# ===============================

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

y_xor = np.array([[0],[1],[1],[0]])

np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

lr = 0.5
epochs = 10000

for _ in range(epochs):

    # Forward
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    output = sigmoid(z2)

    # Backward
    error = y_xor - output

    d2 = error * sigmoid_derivative(output)
    d1 = np.dot(d2, W2.T) * sigmoid_derivative(a1)

    W2 += lr * np.dot(a1.T, d2)
    b2 += lr * np.sum(d2, axis=0, keepdims=True)

    W1 += lr * np.dot(X.T, d1)
    b1 += lr * np.sum(d1, axis=0, keepdims=True)

print("\nXOR Gate Output:")
for i in range(len(X)):
    prediction = output[i][0]   # Proper scalar extraction
    print(X[i], "->", round(prediction))