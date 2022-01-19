# Native

# Site-packages
import numpy as np

# Locals


def MSE(y_true: np.array, y_pred: np.array) -> float:
    return np.round(((y_true - y_pred)**2).mean(), 5)

def MAE(y_true: np.array, y_pred: np.array) -> float:
    return np.round((np.abs(y_true - y_pred)).mean(), 5)

def RMSE(y_true: np.array, y_pred: np.array) -> float:
    return np.round(MSE(y_true, y_pred)**0.5, 5)

def logLoss(y_true: np.array, y_pred: np.array) -> float:
    pred = (y_pred * (1 - 2 * 1e-15)) + 1e-15
    return np.round(-((y_true * np.log(pred)) + ((1 - y_true) * np.log(1 - pred))).mean(axis=0), 5)

def hingeLoss(y_true, y_pred):
    return np.round(np.clip(1 - y_true * y_pred, 0, float('inf')).mean(axis=0), 5)