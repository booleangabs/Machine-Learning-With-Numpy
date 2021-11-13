# Native

# Site-packages
import numpy as np

# Locals
from utils import Algorithm, initializeWeights, Constants
from losses import logLoss, hingeLoss
from activations import Activation, Sigmoid
from preprocessing import Scaler


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
    
class BinaryLogisticRegression(Algorithm):
    def __init__(self, lr: float=1e-3, epochs: int=100):
        self.lr = lr
        self.epochs = epochs
    
    def _sigmoid(self, z):
        return np.round(1 / (1 + np.exp(-z)), 5)
        
    def fit(self, X_train: np.array, y_train: np.array):
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        self.history = {}
        n_rows, n_columns = X_train.shape
        self.W = initializeWeights(n_columns, 1)
        for i in range(self.epochs):
            current_pred = self._sigmoid(X_train.dot(self.W)).flatten()
            diff = (y_train - current_pred)
            dW = X_train.T.dot(diff) * (1 / n_rows)
            self.W += self.lr * dW.reshape(-1, 1)
            
            self.history[i] = logLoss(y_train, current_pred)
        
    def predict_proba(self, X: np.array) -> np.array:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return self._sigmoid(X.dot(self.W)).flatten()
    
    def predict(self, X: np.array, threshold: float=0.5) -> np.array:
        return (self.predict_proba(X) > threshold).astype('int32')
    
class MulticlassLogisticRegression(Algorithm):
    pass

class BinarySVM(Algorithm):
    def __init__(self, C: float=1, lr: float=1e-5, epochs: int=100):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.scaler = Scaler(Constants.PPR_SCALE_MINMAX)
        
    def fit(self, X_train: np.array, y_train_: np.array):
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        y_train = self.scaler.fit_transform(y_train_)
        y_train = y_train * 2 - 1
        self.history = {}
        n_rows, n_columns = X_train.shape
        self.W = np.random.uniform((1 / -n_columns), 1 / n_columns, (n_columns,))
        for i in range(self.epochs):
            current_pred = X_train.dot(self.W)
            prod = y_train * current_pred
            dW_ = -y_train.reshape(-1, 1) * X_train
            dW_[prod >= 1] = 0
            dW = self.W + self.C * dW_.sum(0)
            self.W -= self.lr * dW
            
            self.history[i] = hingeLoss(y_train, current_pred)
            
    def predict_proba(self, X: np.array) -> np.array:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return self.scaler.fit_transform(X.dot(self.W))
        
    def predict(self, X: np.array, threshold: float=0.5) -> np.array:
        probabilities = self.predict_proba(X)
        y_pred = -np.ones_like(probabilities)
        y_pred[probabilities > threshold] = 1
        return y_pred

class SingleLayerPerceptron(Algorithm):
    
    def __init__(self, activation: Activation=Sigmoid, lr: float=1e-5, epochs: int=100):
        self.lr = lr
        self.epochs = epochs
        self.activation = activation()
        
    def fit(self, X_train: np.array, y_train: np.array):
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        self.history = {}
        n_rows, n_columns = X_train.shape
        self.W = np.random.uniform((1 / -n_columns), 1 / n_columns, (n_columns,))
        for i in range(self.epochs):
            current_pred = self.activation(X_train.dot(self.W))
            diff = (current_pred - y_train)
            dW = X_train.T.dot(diff * self.activation.grad(current_pred))
            self.W -= self.lr * dW
            
            self.history[i] = (1 / 2) * (diff**2).sum()
        
    def predict(self, X: np.array) -> np.array:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return self.activation(X.dot(self.W)) 