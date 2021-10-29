# Native

# Site-packages
import numpy as np

# Locals


class Activation:
    def __init__(self, grad: bool=False):
        self.grad = grad
        
    def __call__(self, x):
        return self.__grad(x) if self.grad else self.__call(x)
    
    def __call(self):
        pass
    
    def __grad(self):
        raise NotImplementedError()
        
class Sigmoid(Activation):    
    def __call(self, z):
        return np.round(1 / (1 + np.exp(-z)), 5)
    
    def __grad(self, z):
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))
    

class Relu(Activation):
    def __call(self, x):
        return np.clip(x, 0, float('inf'))
    
    def __grad(self, x):
        grad = np.zeros_like(x)
        grad[x > 0] = 1
        return grad