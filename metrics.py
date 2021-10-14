# Native

# Site-packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Locals


sns.set()

def accuracy(y_true: np.array, y_pred: np.array):
    correct = (y_true == y_pred).sum()
    return np.round(correct / len(y_true), 4)

def recall(y_true: np.array, y_pred: np.array):
    fn = FN(y_true, y_pred)
    tp = TP(y_true, y_pred)
    return tp / (tp + fn)

def falsePositiveRate(y_true: np.array, y_pred: np.array):
    fp = FP(y_true, y_pred)
    tn = TN(y_true, y_pred)
    return fp / (fp + tn)

def precision(y_true: np.array, y_pred: np.array):
    tp = TP(y_true, y_pred)
    fp = FP(y_true, y_pred)
    return tp / (tp + fp)

def f1Score(y_true: np.array, y_pred: np.array):
    return 2 / ((1 / precision(y_true, y_pred)) + (1 / recall(y_true, y_pred)))

def roc(y_true: np.array, probabilities: np.array, n_thresholds: float) -> tuple:
    tpr = []
    fpr = []
    for i in range(n_thresholds + 1):
        threshold = i / n_thresholds
        y_pred = (probabilities >= threshold).astype('uint8')
        tpr.append(recall(y_true, y_pred))
        fpr.append(falsePositiveRate(y_true, y_pred))
    return np.array(tpr), np.array(fpr)

def plotROC(tpr: np.array, fpr: np.array, n_thresholds: int):
    figure = plt.figure()
    axis = figure.add_subplot()
    axis.set_title(f"ROC - AUC = {AUCROC(tpr, fpr, n_thresholds)}")
    plt.scatter(fpr, tpr)
    plt.show()

def AUCROC(tpr: np.array, fpr: np.array, n_thresholds: int) -> float:
    sum_ = 0
    for i in range(n_thresholds):
        sum_ += (fpr[i] - fpr[i + 1]) * tpr[i]
    return np.round(sum_, 3)

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