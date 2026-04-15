import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
x = np.random.uniform(-10000, 10000, 100)

x = np.sort(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1):
    return np.where(x > 0, 1, alpha * np.exp(x))

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_derivative(x):
    return sigmoid(x)


activations = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("Tanh", tanh, tanh_derivative),
    ("ReLU", relu, relu_derivative),
    ("Leaky ReLU", leaky_relu, leaky_relu_derivative),
    ("ELU", elu, elu_derivative),
    ("Softplus", softplus, softplus_derivative),
]

for name, func, derivative in activations:
    
    plt.figure(figsize=(10,4))
    
    # Activation Plot
    plt.subplot(1,2,1)
    plt.plot(x, func(x))
    plt.title(f"{name} Function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    
    # Derivative Plot
    plt.subplot(1,2,2)
    plt.plot(x, derivative(x))
    plt.title(f"{name} Derivative")
    plt.xlabel("x")
    plt.ylabel("f'(x)")
    
    plt.tight_layout()
    plt.show()
