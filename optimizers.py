import numpy as np

# Stochastic Gradient Descent
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        for param in self.parameters:
            param["value"] -= self.lr * param["grad"]

# Adam Optimizer
class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p["value"]) for p in parameters]
        self.v = [np.zeros_like(p["value"]) for p in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.paramters):
            gradients = param["grad"]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.m[i] / (1 - self.beta2 ** self.t)
            param["value"] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# AdamW Optimizer with Weight Decay
class AdamW(Adam):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, beta1, beta2, epsilon)
        self.weight_decay = weight_decay

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            gradients = param["grad"] + self.weight_decay * param["value"]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param["value"] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# RMSprop Optimizer
class RMSprop:
    def __init__(self, parameters, lr=0.001, beta=0.9, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.v = [np.zeros_like(p["value"]) for p in parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            gradients = param["grad"]
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (gradients ** 2)
            param["value"] -= self.lr * gradients / (np.sqrt(self.v[i]) + self.epsilon)