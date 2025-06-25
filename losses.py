import numpy as np
# Mean Squared Error Loss Function and Gradient
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

def cross_entropy_grad(y_pred, y_true):
    return (y_pred - y_true) / y_true.shape[0]