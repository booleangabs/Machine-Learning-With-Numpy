# Native

# Site-packages
import numpy as np

# Locals
from losses import MSE
from utils import Algorithm

        
class LinearRegression(Algorithm):
    def __init__(self):
        pass
        
    def fit(self, X_train: np.array, y_train: np.array):
        X_train_1 = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        moore_penrose = np.linalg.inv(X_train_1.T @ X_train_1) @ X_train_1.T
        self.W = moore_penrose @ y_train
        self.train_score = MSE(y_train, self.predict(X_train))
            
    def predict(self, X: np.array) -> np.array:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X.dot(self.W)

class GradientDescentRegression(Algorithm):
    def __init__(self, alpha: float=1e-5, epochs: int=100):
        self.alpha = alpha
        self.epochs = epochs
        
    def fit(self, X_train: np.array, y_train: np.array):
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        self.history = {}
        n_rows, n_columns = X_train.shape
        self.W = np.random.uniform((1 / -n_columns), 1 / n_columns, (n_columns,))
        for i in range(self.epochs):
            current_pred = X_train.dot(self.W)
            diff = (current_pred - y_train)
            dW = X_train.T.dot(diff) * (2 / n_rows)
            self.W -= self.alpha * dW
            
            self.history[i] = MSE(y_train, current_pred)
        
    def predict(self, X: np.array) -> np.array:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X.dot(self.W)
    
class LassoRegression(Algorithm):
    def __init__(self, alpha: float=1e-5, lambda_: float=5e-2, epochs: int=100):
        self.alpha = alpha
        self.epochs = epochs
        
    def fit(self, X_train: np.array, y_train: np.array):
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        self.history = {}
        n_rows, n_columns = X_train.shape
        self.W = np.random.uniform((1 / -n_columns), 1 / n_columns, (n_columns,))
        for i in range(self.epochs):
            current_pred = X_train.dot(self.W)
            diff = (current_pred - y_train)
            dW = X_train.T.dot(diff) * (2 / n_rows) + np.sign(self.W)
            self.W -= self.alpha * dW
            
            W_norm = np.linalg.norm(self.W)
            self.history[i] = ((y_train - current_pred)**2 + W_norm).mean()
        
    def predict(self, X: np.array) -> np.array:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X.dot(self.W)