# Native

# Site-packages 
import numpy as np

# Locals
from classification import KNN

class KMeans:
    def __init__(self, n_clusters: int=-1):
        self.n_clusters = n_clusters
        self.labels = None
        self.inertia = None
        
    def fit(self, X: np.array, iterations: int=1000):
        # Mu = initialize centers with kmeans++
        # 
        # for i=0,1,...,iterations-1 do
        #     labels = assignClusters(Mu)
        #     Mu = updateMeans(X, labels)
        #     np.append(self.inertia, calculateCurrInertia(X, Mu)
        
        pass
    
    def predict(self, X_new: np.array, n_neighbours: int=3):
        pass
    
    class SimpleKneeDetector:
        def __init__(self, inertia: np.array, max_n: int=20):
            self.inertia = inertia
            self.max_n = max_n
            
        def __call__(self):
            for i in range(2, self.max_n - 1):
                p = (self.inertia[i] / self.inertia[i + 1]) - (self.inertia[i] / self.inertia[i - 1])
                if 0 <= p <= 1:
                    break
            return i + 1
    