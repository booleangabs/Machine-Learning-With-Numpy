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
        
def initializeWeights(n_inputs, n_outputs):
    abs_n = 1 / n_inputs
    return np.random.uniform(-abs_n, abs_n, (n_inputs, n_outputs))