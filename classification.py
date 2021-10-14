# Native

# Site-packages
import numpy as np

# Locals
from utils import Algorithm

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
        return preds
        
    def __computeDist(self, d: np.array, X: np.array):
        def dist(x, y):
            return np.sqrt((x - y).T.dot(x - y))
        return np.float32([dist(d, i) for i in X])