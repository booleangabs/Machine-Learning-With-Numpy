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
        
class Activation:
    def __init__(self):
        pass
    
    def __call__(self):
        raise NotImplementedError()
    
    def grad(self):
        raise NotImplementedError()
        
class Sigmoid(Activation):
    def __init__(self):
        pass
    

class Relu(Activation):
    def __init__(self):
        pass