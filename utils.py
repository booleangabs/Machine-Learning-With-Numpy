import numpy as np

class Constants:
    # Preprocessing (ppr)
    PPR_SCALE_MINMAX = 0
    PPR_SCALE_STD = 1
    
class Algorithm:
    def __init__(self):
        pass
    
    def fit(self):
        raise NotImplementedError()
    
    def predict(self):
        raise NotImplementedError()
        
class Activation:
    def __init__(self, grad: bool=False):
        self.grad = grad
        
    def __call__(self):
        raise NotImplementedError()
    
    def __grad(self):
        raise NotImplementedError()
        
class Sigmoid(Activation):
    def __call__(self, z):
        return self.__grad(z) if self.grad else self.__sigmoid(z)
    
    def __sigmoid(self, z):
        return np.round(1 / (1 + np.exp(-z)), 5)
    
    def __grad(self, z):
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))
    

class Relu(Activation):
    def __call__(self, x):
        return self.__grad(x) if self.grad else self.__relu(x)
    
    def __relu(self, x):
        return np.clip(x, 0, float('inf'))
    
    def __grad(self, x):
        grad = np.zeros_like(x)
        grad[x > 0] = 1
        return grad