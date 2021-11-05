# Native

# Site-packages
import numpy as np

# Locals


class Activation:
    def __init__(self):
        pass
        
    def __call__(self):
        raise NotImplementedError()
    
    def grad(self):
        raise NotImplementedError()
        
class Sigmoid(Activation):
    def __call__(self, z):
        return np.round(1 / (1 + np.exp(-z)), 5)
    
    def grad(self, z):
        return self(z) * (1 - self(z))
    
class ReLU(Activation):
    def __call__(self, x):
        return np.clip(x, 0, float('inf'))
    
    def grad(self, x):
        grad = np.zeros_like(x)
        grad[x > 0] = 1
        return grad