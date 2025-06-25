import numpy as np

# ReLU Activation Function
class ReLU:
    def forward(self, x):
        """ReLU(x) = max(0, x)"""
        self.x = x
        return np.maximum(0,x)
    
    def backward(self, grad):
        """ReLU'(x) = 1 if x > 0, else 0"""
        return grad * (self.x > 0)
    
#  Sigmoid Activation Function
class Sigmoid:
    def forward(self, x):
        """sigmoid(x) = 1 / (1 + e^(-x))"""
        self.sigmoid = 1 / (1 + np.exp(-x))
        return self.sigmoid
    
    def backward(self, grad):
        """sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))"""
        return grad * self.sigmoid * (1 - self.sigmoid)

# Tanh Activation Function
class Tanh:
    def forward(self, x):
        """tanh(x)"""
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        """tanh'(x) = 1 - tanh^2(x)"""
        return grad * (1 - self.out**2)

# Swish Activation Function
class Swish:
    def forward(self, x):
        """Swish(x) = x * sigmoid(x)"""
        self.sigmoid = 1 / (1 + np.exp(-x))
        self.x = x
        return x * self.sigmoid

    def backward(self, grad):
        """Swish'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
        return grad * (self.sigmoid + self.x * self.sigmoid * (1 - self.sigmoid))
    