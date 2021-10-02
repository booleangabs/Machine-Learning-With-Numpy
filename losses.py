import numpy as np

def MSE(y_true: np.array, y_pred: np.array) -> float:
    return np.round(((y_true - y_pred)**2).mean(axis=0), 5)