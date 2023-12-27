"""
MIT License

Copyright (c) 2023 Gabriel Tavares (booleangabs)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# site-packages
import numpy as np

# local
from ..__common import BaseModel


LIN_REG_SOLVERS = ["pinv", "gradient_descent", "sgd"]


class LinearRegression(BaseModel):
    def __init__(self, 
                 solver: str = "pinv", 
                 name: str = "linear_regressor", 
                 max_iter: int = 1000,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 random_state: int = 0,
                ):
        assert solver in LIN_REG_SOLVERS, f"Unknown solver '{solver}'. Options are {LIN_REG_SOLVERS}."
        self.solver = solver
        self.name = name
        super().__init__(name)
        self.weights = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.__generator = np.random.default_rng(self.random_state)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.solver == "pinv":
            self.__solve_pinv(X_train, y_train)
        elif self.solver == "gradient_descent":
            self.__solve_gradient_descent(X_train, y_train)
        else:
            self.__solve_sgd(X_train, y_train)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert type(self.weights) == np.ndarray, "Model has not been trained yet!"
        return X @ self.weights
            
    def __solve_pinv(self, X_train: np.ndarray, y_train: np.ndarray):
        self.weights = np.linalg.pinv(X_train) @ y_train
        
    def __solve_gradient_descent(self, X_train: np.ndarray, y_train: np.ndarray):
        self.weights = self.__generator.standard_normal((X_train.shape[1], 1))
        for i in range(self.max_iter):
            diff = (y_train - self.predict(X_train))
            grad_w = np.expand_dims(-2 * (diff * X_train).mean(0), -1)            
            self.weights -= self.learning_rate * grad_w
    
    def __solve_sgd(self, X_train: np.ndarray, y_train: np.ndarray):
        self.weights = self.__generator.standard_normal((X_train.shape[1], 1))
        N = len(X_train)
        for i in range(self.max_iter):
            idxs = np.random.randint(0, N, self.batch_size)
            X_ = X_train[idxs]
            y_ = y_train[idxs]
            diff = (y_ - self.predict(X_))
            grad_w = np.expand_dims(-2 * (diff * X_).mean(0), -1)
            self.weights -= self.learning_rate * grad_w
            
