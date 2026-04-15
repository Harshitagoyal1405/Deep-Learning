import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

X_train = load_images("dataset-4/train-images.idx3-ubyte")
y_train = load_labels("dataset-4/train-labels.idx1-ubyte")

X_test = load_images("dataset-4/t10k-images.idx3-ubyte")
y_test = load_labels("dataset-4/t10k-labels.idx1-ubyte")

# Use subset for faster training
X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = X_test[:2000]
y_test = y_test[:2000]

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encoding
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_onehot = one_hot(y_train)
y_test_onehot = one_hot(y_test)



def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)



np.random.seed(42)

input_size = 784
hidden_size = 128
output_size = 10

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.01
epochs = 20
batch_size = 128



for epoch in range(epochs):

    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train_onehot = y_train_onehot[permutation]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train_onehot[i:i+batch_size]

        # Forward
        z1 = np.dot(X_batch, W1) + b1
        a1 = relu(z1)

        z2 = np.dot(a1, W2) + b2
        output = softmax(z2)

        # Backward
        d2 = output - y_batch
        dW2 = np.dot(a1.T, d2) / batch_size
        db2 = np.sum(d2, axis=0, keepdims=True) / batch_size

        d1 = np.dot(d2, W2.T) * relu_derivative(z1)
        dW1 = np.dot(X_batch.T, d1) / batch_size
        db1 = np.sum(d1, axis=0, keepdims=True) / batch_size

        # Update
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Evaluate each epoch
    z1_test = np.dot(X_test, W1) + b1
    a1_test = relu(z1_test)
    z2_test = np.dot(a1_test, W2) + b2
    test_output = softmax(z2_test)

    y_pred = np.argmax(test_output, axis=1)
    acc = accuracy_score(y_test, y_pred)

    print(f"Epoch {epoch+1}, Accuracy: {acc:.4f}")

print("\nFinal Test Accuracy:", acc)
