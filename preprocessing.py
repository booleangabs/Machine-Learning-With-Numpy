# Native
import warnings
warnings.filterwarnings('ignore')

# Site packages
import numpy as np

# Locals
from utils import Constants as cts


class Scaler:
    def __init__(self, mode: int):
        assert mode in (0, 1), f"{mode} is not a valid value for mode."
        self.mode = mode
        
    def transform(self, X: np.array):
        if self.mode == cts.PPR_SCALE_MINMAX:
            X_max = X.max(0)
            X_min = X.min(0)
            X_transformed = np.true_divide((X - X_min), (X_max - X_min))
        else:
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                X_transformed = np.true_divide(X - X_mean, X_std)
                X_transformed[X_transformed == np.inf] = 0
        return np.nan_to_num(X_transformed, nan=0.0, posinf=0, neginf=0)
            