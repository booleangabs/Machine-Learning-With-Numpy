import numpy as np

def accurary(y_true: np.array, y_pred: np.array):
    correct = (y_true == y_pred).sum()
    return np.round(correct / len(y_true), 4)

def recall(y_true: np.array, y_pred: np.array):
    pass

def precision(y_true: np.array, y_pred: np.array):
    pass

def f1Score(y_true: np.array, y_pred: np.array):
    pass

def confusionMatrix(y_true: np.array, y_pred: np.array):
    pass

def plotConfusionMatrix(y_true: np.array, y_pred: np.array):
    pass