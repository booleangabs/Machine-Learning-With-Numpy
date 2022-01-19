# Native
import warnings
warnings.filterwarnings('ignore')

# Site packages
import numpy as np

# Locals


class Kernel:
    def __init__(self):
        pass
    
    def transform(self):
        raise NotImplementedError()
    
    def inverse_transform(self):
        raise NotImplementedError()