import numpy as np
from losses import MSE

class Regression:
    def __init__(self):
        pass
    
    def fit(self):
        raise NotImplementedError()
    
    def predict(self):
        raise NotImplementedError()
        
        
class SimpleLinearRegression(Regression):
    def __init__(self, mode: str):
        assert mode in ('Gradient Descent', 'Pseudo Inverse')
        self.mode = mode
        
    def fit(self, X_train: np.array, y_train: np.array, alpha=None, max_iter=1000, patience=5):
        if self.mode == "Gradient Descent":
            assert (alpha != 0) & (0 < alpha)
            self.history = {}
            self.w0, self.w1 = np.random.randn(2)
            current_pred = np.zeros_like(y_train)
            last_loss = float('inf')
            ticker = 0
            for i in range(max_iter):
                if ticker > patience:
                    break
                current_pred = self.w0 * X_train + self.w1
                diff = (y_train - current_pred)
                dw0 = -2 * np.round((diff * X_train).mean(axis=0), 4)
                dw1 = -2 * diff.mean(axis=0)
                self.w0 -= alpha * dw0
                self.w1 -= alpha * dw1
                
                current_loss = MSE(y_train, current_pred)
                self.history[i] = current_loss
                if abs(current_loss - last_loss) < 0.1: # Early stopping
                    ticker += 1 
                last_loss = current_loss
        else:
            moore_penrose = np.linalg.inv(X_train.T @ X_train) @ X_train.T
            self.w = moore_penrose @ y_train
            
    def predict(self, X: np.array) -> np.array:
        if self.mode == "Gradient Descent":
            prediction = self.w0*X.flatten() + self.w1
        else:
            prediction = (self.w * X).sum(axis=1)
        return prediction
            