import numpy as np
class Dense:
    def __init__(self, in_features, out_features):
        # in_features: number of input features to the layer
        # out_features: number of output features (neurons) in the layer
        self.W = np.random.randn(in_features, out_features) * 0.01 # weight matrix
        self.b = np.zeros((1, out_features)) # bias vector

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, gradient):
        self.dw = np.dot(self.x.T, gradient) # gradient wrt weights
        self.db = np.sum(gradient, axis=0, keepdims=True) # gradient wrt bias
        self.dx = np.dot(gradient, self.W.T) # gradient to propagte backward
        return self.dx
    
    def parameters(self):
        return [{"value": self.W, "grad": self.dw}, {"vale": self.b, "grad": self.db}]