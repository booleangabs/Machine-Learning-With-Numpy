# Native

# Site-packages
import numpy as np

# Locals


def MSE(y_true: np.array, y_pred: np.array) -> float:
    return np.round(((y_true - y_pred)**2).mean(axis=0), 5)

def MAE(y_true: np.array, y_pred: np.array) -> float:
    return np.round((np.abs(y_true - y_pred)).mean(axis=0), 5)

def RMSE(y_true: np.array, y_pred: np.array) -> float:
    return MSE(y_true, y_pred)**0.5