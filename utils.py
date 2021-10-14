import numpy as np

class Constants:
    # Preprocessing (ppr)
    PPR_SCALE_MINMAX = 0
    PPR_SCALE_STD = 1
    
class Algorithm:
    def __init__(self):
        pass
    
    def fit(self):
        raise NotImplementedError()
    
    def predict(self):
        raise NotImplementedError()