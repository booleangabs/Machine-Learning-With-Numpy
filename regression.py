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
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        moore_penrose = np.linalg.inv(X_train.T @ X_train) @ X_train.T
        self.W = moore_penrose @ y_train
        self.train_score = MSE(y_train, self.predict(X_train))
            
    def predict(self, X: np.array) -> np.array:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X.dot(self.W)