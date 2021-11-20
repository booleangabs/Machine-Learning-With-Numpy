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
        
class L1(Regularizer):
    def __call__(self, X: np.array, lambda_: float) -> float:
        X_norm = np.linalg.norm(X)
        return lambda_ * X_norm
    
    def grad(self, X: np.array, lambda_: float) -> float:
        return lambda_ * np.sign(X)
        
class L2(Regularizer):
    def __call__(self, X: np.array, omega_: float) -> float:
        X_norm_squared = np.linalg.norm(X) ** 2
        return omega_ * X_norm_squared
    
    def grad(self, X: np.array, omega_: float) -> float:
        return 2 * omega_ * X