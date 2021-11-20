# Native

# Site-packages
import numpy as np

# Locals


class Regularizer:
    def __init__(self):
        pass
        
    def __call__(self):
        raise NotImplementedError()
    
    def grad(self):
        raise NotImplementedError()
        
class Placeholder(Regularizer):
    def __call__(self, X: np.array) -> float:
        return 0
    
    def grad(self, X: np.array) -> np.array:
        return np.zeros(X.shape[0])
        
class L1(Regularizer):
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_
        
    def __call__(self, X: np.array) -> float:
        X_norm = np.abs(X).sum()
        return self.lambda_ * X_norm
    
    def grad(self, X: np.array) -> np.array:
        return self.lambda_ * np.sign(X)
        
class L2(Regularizer):
    def __init__(self, omega_: float):
        self.omega_ = omega_    

    def __call__(self, X: np.array) -> float:
        X_norm_squared = np.linalg.norm(X) ** 2
        return self.omega_ * X_norm_squared
    
    def grad(self, X: np.array) -> np.array:
        return 2 * self.omega_ * X
    
class ElasticNet(Regularizer):
    def __init__(self, lambda_1: float, lambda_2: float):
        self.l1 = L1(1)
        self.l2 = L2(1)
        self.alpha = lambda_2 / (lambda_1 + lambda_2)
        
    def __call__(self, X: np.array) -> float:
        return self.alpha * self.l2(X) + (1 - self.alpha) * self.l1(X)
        
    def grad(self, X: np.array) -> np.array:
        return self.alpha * self.l2.grad(X) + (1 - self.alpha) * self.l1.grad(X)