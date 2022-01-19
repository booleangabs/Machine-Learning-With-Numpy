# Native
import warnings
warnings.filterwarnings('ignore')

# Site packages
import numpy as np

# Locals
from utils import Constants as cts


class Preprocessing:
    def __init__(self):
        pass
        
    def fit(self):
        raise NotImplementedError()
        
    def transform(self, X: np.array) -> np.array:
        raise NotImplementedError()

    def fit_transform(self, X: np.array) -> np.array:
        raise NotImplementedError()

class Scaler(Preprocessing):
    def __init__(self, mode: int):
        assert mode in (0, 1), f"{mode} is not a valid value for mode."
        self.mode = mode
        
    def fit(self, X: np.array):
        if self.mode == cts.PPR_SCALE_MINMAX:
            self.X_max = X.max(0)
            self.X_min = X.min(0)
        else:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
        
    def transform(self, X: np.array) -> np.array:
        if self.mode == cts.PPR_SCALE_MINMAX:
            X_transformed = np.true_divide((X - self.X_min), (self.X_max - self.X_min))
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                X_transformed = np.true_divide(X - self.X_mean, self.X_std)
                X_transformed[X_transformed == np.inf] = 0
        return np.nan_to_num(X_transformed, nan=0.0, posinf=0, neginf=0)

    def fit_transform(self, X: np.array) -> np.array:
        self.fit(X)
        return self.transform(X)

class CategoricalEncoder(Preprocessing):
    def __init__(self, mode: int):
        assert mode in (0, 1), f"{mode} is not a valid value for mode."
        self.mode = mode
    
    def encode(self, X: np.array) -> np.array:
        if self.mode == cts.ENC_CAT2ONEHOT:
            self.unique = np.unique(X)
            unique_list = self.unique.tolist()
            result = np.zeros((len(X), len(self.unique)))
            for i, x in enumerate(X):
                result[i][unique_list.index(x)] = 1
            return result
        return np.argmax(X, axis=1)
    
    def reverse_encode(self, X: np.array):
        if self.mode == cts.ENC_CAT2ONEHOT:
            return np.argmax(X, axis=1)
        self.unique = list(np.unique(X))
        result = np.zeros((len(X), len(self.unique)))
        for i, x in enumerate(X):
            result[i][self.unique.index(x)] = 1
        return result
                    
def textToCategorical(X: np.array) -> np.array:
    return np.argmax(CategoricalEncoder(0).encode(X), axis=1), np.unique(X)

def categoricalToText(X: np.array, unique: np.array) -> np.array:
    result = []
    for x in X:
        result.append(unique[x])
    return np.array(result)