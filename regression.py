# Native

# Site-packages
import numpy as np

# Locals
from losses import MSE
from utils import Algorithm
from regularizers import Regularizer, Placeholder, L1, L2, ElasticNet

        
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
    def __init__(self, lr: float=1e-5, epochs: int=100, regularizer: Regularizer=None):
        self.lr = lr
        self.epochs = epochs
        if not regularizer:
            self.regularizer = Placeholder()
        else:
            self.regularizer = regularizer
            
    def fit(self, X_train: np.array, y_train: np.array):
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        self.history = {}
        n_rows, n_columns = X_train.shape
        self.W = np.random.uniform((1 / -n_columns), 1 / n_columns, (n_columns,))
        for i in range(self.epochs):
            current_pred = X_train.dot(self.W)
            diff = (current_pred - y_train)
            dW = X_train.T.dot(diff) * (2 / n_rows)
            self.W -= self.lr * dW + self.regularizer.grad(self.W)

            self.history[i] = ((y_train - current_pred)**2 + self.regularizer(self.W)).mean()
        
    def predict(self, X: np.array) -> np.array:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X.dot(self.W)

class LassoRegression(GradientDescentRegression):
    def __init__(self, lambda_: float, lr: float=1e-5, epochs: int=100):
        self.regularizer = L1(lambda_)
        self.lambda_ = lambda_
        super(LassoRegression, self).__init__(lr=lr, 
                                              epochs=epochs, 
                                              regularizer=self.regularizer)
        
class RidgeRegression(GradientDescentRegression):
    def __init__(self, omega_: float, lr: float=1e-5, epochs: int=100):
        self.regularizer = L2(omega_)
        self.omega_ = omega_
        super(RidgeRegression, self).__init__(lr=lr, 
                                              epochs=epochs, 
                                              regularizer=self.regularizer)
        
class ElasticNetRegression(GradientDescentRegression):
    def __init__(self, lambda_1: float, lambda_2: float, lr: float=1e-5, epochs: int=100):
        self.regularizer = ElasticNet(lambda_1, lambda_2)
        self.lambdas = (lambda_1, lambda_2)
        super(ElasticNetRegression, self).__init__(lr=lr, 
                                              epochs=epochs, 
                                              regularizer=self.regularizer)