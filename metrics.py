import numpy as np

def accuracy(y_true: np.array, y_pred: np.array):
    correct = (y_true == y_pred).sum()
    return np.round(correct / len(y_true), 4)

def recall(y_true: np.array, y_pred: np.array):
    fn = FN(y_true, y_pred)
    tp = TP(y_true, y_pred)
    return tp / (tp + fn)

def precision(y_true: np.array, y_pred: np.array):
    tp = TP(y_true, y_pred)
    fp = FP(y_true, y_pred)
    return tp / (tp + fp)

def f1Score(y_true: np.array, y_pred: np.array):
    return 2 / ((1 / precision(y_true, y_pred)) + (1 / recall(y_true, y_pred)))

def confusionMatrix(y_true: np.array, y_pred: np.array):
    pass

def plotConfusionMatrix(y_true: np.array, y_pred: np.array):
    pass

def TP(y_true: np.array, y_pred: np.array) -> int:
    return np.bitwise_and(y_true == 1, y_pred == 1).sum()

def TN(y_true: np.array, y_pred: np.array) -> int:
    return np.bitwise_and(y_true == 0, y_pred == 0).sum()

def FP(y_true: np.array, y_pred: np.array) -> int:
    return np.bitwise_and(y_true == 0, y_pred == 1).sum()

def FN(y_true: np.array, y_pred: np.array) -> int:
    return np.bitwise_and(y_true == 1, y_pred == 0).sum()