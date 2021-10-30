# Native

# Site-packages
import numpy as np

# Locals
from utils import Algorithm
from losses import logLoss


class KNN(Algorithm):
    def __init__(self, k: int=3):
        self.k = k
        
    def fit(self, X_train: np.array, y_train: np.array):
        self.X = X_train
        self.y = y_train
        
    def predict(self, X: np.array):
        preds = []
        for new_point in X:
            distances = self.__computeDist(new_point, self.X)
            indices = np.argsort(distances)
            k_nearest_y = self.y[indices][:self.k]
            values, counts = np.unique(k_nearest_y, return_counts=True)
            idx = counts.tolist().index(counts.max())
            mode = values[idx]
            preds.append(mode)
        return np.array(preds)
        
    def __computeDist(self, d: np.array, X: np.array):
        def dist(x, y):
            return np.sqrt((x - y).T.dot(x - y))
        return np.float32([dist(d, i) for i in X])
    
class LogisticRegression(Algorithm):
    def __init__(self):
        pass
    
    def _sigmoid(self, z):
        return np.round(1 / (1 + np.exp(z)), 5)
    
    def fit(self, X_train: np.array, y_train: np.array, alpha=1e-3, epochs=100):
        assert (0 < alpha <= 1)
        self.history = {}
        n_inputs = X_train.shape[1]
        self.b, self.W = 0, np.random.uniform((1 / -n_inputs), 1 / n_inputs, (n_inputs,))
        current_pred = np.zeros_like(y_train)
        for i in range(epochs):
            z = (X_train.dot(self.W) + self.b)
            current_pred = self._sigmoid(z)
            diff = -(y_train - current_pred)
            dW = -(diff.dot(X_train))
            db = -diff.mean(0)
            self.W -= alpha * dW
            self.b -= alpha * db
            
            self.history[i] = logLoss(y_train, current_pred)
        
    def predict(self, X: np.array) -> np.array:
        prediction = self._sigmoid(X.dot(self.W) + self.b)
        return prediction