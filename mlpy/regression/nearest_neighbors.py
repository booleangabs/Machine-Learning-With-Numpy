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


KNN_REDUCTIONS = ["mean", "median"]


class KnnRegressor(BaseModel):
    def __init__(self, 
                 name: str = "knn_regressor",
                 k: int = 3,
                 reduction: str = "mean"
                ):
        assert reduction in KNN_REDUCTIONS, "Invalid reduction mode: {reduction}. Options are {KNN_REDUCTIONS}"
        assert k > 0, "The number of neighbors 'k' must be greater than 0. Got k = {k}."
        self.name = name
        super().__init__(name)
        self.k = k
        self.reduction = reduction
        if self.reduction == "mean":
            self.rdfunc = np.mean
        else:
            self.rdfunc = np.median
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert type(self.y_train) == np.ndarray, "Model has not been trained yet!"
        predictions = np.zeros((X.shape[0], 1))
        for i, x in enumerate(X):
            x_ = x[np.newaxis, ...]
            distances = np.sum((self.X_train - x_)**2, axis=1)
            ids = np.argsort(distances)
            nn = self.y_train[ids][:self.k]
            predictions[i] = self.rdfunc(nn)
        return predictions
