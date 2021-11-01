# Native

# Site-packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Locals


sns.set()

def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    correct = (y_true == y_pred).sum()
    return np.round(correct / len(y_true), 4)

def recall(y_true: np.ndarray, y_pred: np.ndarray):
    fn = FN(y_true, y_pred)
    tp = TP(y_true, y_pred)
    return tp / (tp + fn)

def falsePositiveRate(y_true: np.ndarray, y_pred: np.ndarray):
    fp = FP(y_true, y_pred)
    tn = TN(y_true, y_pred)
    return fp / (fp + tn)

def precision(y_true: np.ndarray, y_pred: np.ndarray):
    tp = TP(y_true, y_pred)
    fp = FP(y_true, y_pred)
    return tp / (tp + fp)

def f1Score(y_true: np.ndarray, y_pred: np.ndarray):
    return 2 / ((1 / precision(y_true, y_pred)) + (1 / recall(y_true, y_pred)))

def roc(y_true: np.ndarray, probabilities: np.ndarray, n_thresholds: float) -> tuple:
    tpr = []
    fpr = []
    for i in range(n_thresholds + 1):
        threshold = i / n_thresholds
        y_pred = (probabilities >= threshold).astype('uint8')
        tpr.append(recall(y_true, y_pred))
        fpr.append(falsePositiveRate(y_true, y_pred))
    return np.array(tpr), np.array(fpr)

def plotROC(tpr: np.ndarray, fpr: np.ndarray, n_thresholds: int):
    figure = plt.figure()
    axis = figure.add_subplot()
    axis.set_ylim([0 - 1e-1, 1 + 1e-1])
    axis.set_xlim([0 - 1e-1, 1 + 1e-1])
    axis.set_title(f"ROC - AUC = {AUCROC(tpr, fpr, tpr.shape[0] - 1)}")
    plt.scatter(fpr, tpr)
    plt.show()

def AUCROC(tpr: np.ndarray, fpr: np.ndarray, n_thresholds: int) -> float:
    sum_ = 0
    for i in range(n_thresholds):
        sum_ += (fpr[i] - fpr[i + 1]) * tpr[i]
    return np.round(sum_, 5)

def confusionMatrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool=False) -> np.ndarray:
    n_classes = len(np.unique(y_true))
    cm = np.zeros((n_classes, n_classes), 'int')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm[i][j] = int(((y_true == i) & (y_pred == j)).sum())
    if normalize:
        cm = cm / cm.sum()
    return cm

def plotConfusionMatrix(y_true: np.ndarray, y_pred: np.ndarray):
    sns.heatmap(confusionMatrix(y_true, y_pred), annot=True, fmt='d')

def TP(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return np.bitwise_and(y_true == 1, y_pred == 1).sum()

def TN(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return np.bitwise_and(y_true == 0, y_pred == 0).sum()

def FP(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return np.bitwise_and(y_true == 0, y_pred == 1).sum()

def FN(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return np.bitwise_and(y_true == 1, y_pred == 0).sum()