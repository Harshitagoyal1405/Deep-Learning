import numpy as np
import struct
from sklearn.metrics import accuracy_score

# ============================
# LOAD MNIST
# ============================

def load_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.fromfile(f, dtype=np.uint8)
        return data.reshape(num, rows*cols)

def load_labels(filename):
    with open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.fromfile(f, dtype=np.uint8)

X_train = load_images("dataset-4/train-images.idx3-ubyte")[:10000] / 255.0
y_train = load_labels("dataset-4/train-labels.idx1-ubyte")[:10000]

X_test = load_images("dataset-4/t10k-images.idx3-ubyte")[:2000] / 255.0
y_test = load_labels("dataset-4/t10k-labels.idx1-ubyte")[:2000]

def one_hot(y, classes=10):
    return np.eye(classes)[y]

y_train = one_hot(y_train)

# ============================
# ACTIVATIONS
# ============================

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ============================
# TRAIN FUNCTION
# ============================

def train_model(use_momentum=False, use_l2=False):

    np.random.seed(42)

    input_size = 784
    hidden = 128
    output_size = 10

    W1 = np.random.randn(input_size, hidden) * np.sqrt(2/input_size)
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, output_size) * np.sqrt(2/hidden)
    b2 = np.zeros((1, output_size))

    vW1 = np.zeros_like(W1)
    vW2 = np.zeros_like(W2)

    lr = 0.01
    epochs = 30
    batch_size = 128
    beta = 0.9
    lambda_l2 = 0.001

    for epoch in range(epochs):

        perm = np.random.permutation(len(X_train))
        X_shuff = X_train[perm]
        y_shuff = y_train[perm]

        for i in range(0, len(X_train), batch_size):

            Xb = X_shuff[i:i+batch_size]
            yb = y_shuff[i:i+batch_size]

            z1 = np.dot(Xb, W1) + b1
            a1 = relu(z1)
            z2 = np.dot(a1, W2) + b2
            out = softmax(z2)

            d2 = out - yb
            dW2 = np.dot(a1.T, d2)/batch_size
            db2 = np.sum(d2, axis=0, keepdims=True)/batch_size

            d1 = np.dot(d2, W2.T) * relu_derivative(z1)
            dW1 = np.dot(Xb.T, d1)/batch_size
            db1 = np.sum(d1, axis=0, keepdims=True)/batch_size

            if use_l2:
                dW2 += lambda_l2 * W2
                dW1 += lambda_l2 * W1

            if use_momentum:
                vW2 = beta*vW2 + (1-beta)*dW2
                vW1 = beta*vW1 + (1-beta)*dW1
                W2 -= lr * vW2
                W1 -= lr * vW1
            else:
                W2 -= lr * dW2
                W1 -= lr * dW1

            b2 -= lr * db2
            b1 -= lr * db1

    z1 = np.dot(X_test, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    test_out = softmax(z2)
    y_pred = np.argmax(test_out, axis=1)

    return accuracy_score(y_test, y_pred)

# ============================
# RUN EXPERIMENTS
# ============================

acc_sgd = train_model(False, False)
acc_l2 = train_model(False, True)
acc_mom = train_model(True, False)
acc_mom_l2 = train_model(True, True)

print("SGD Accuracy:", acc_sgd)
print("SGD + L2 Accuracy:", acc_l2)
print("Momentum Accuracy:", acc_mom)
print("Momentum + L2 Accuracy:", acc_mom_l2)