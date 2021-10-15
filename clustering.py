# Native

# Site-packages 
import numpy as np

# Locals
from classification import KNN

class KMeans:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.inertia = float('inf')
        
    def fit(self, X: np.array, iterations: int=1000):
        self.labels = np.zeros((X.shape[0],)) - 1
        self.__initCentroids(X, self.n_clusters)
        
        last = self.labels.copy()
        for _ in range(iterations):
            means = np.zeros((self.n_clusters, X.shape[1]))
            counts = np.zeros((self.n_clusters,))
            for idx, x in enumerate(X):
                argmin_dist = np.argmin([self.__distance(x, i) for i in self.centroids])
                self.labels[idx] = argmin_dist
                means[argmin_dist] += x
                counts[argmin_dist] += 1
            means = (means.T / counts).T
            self.centroids = means
            if last == self.labels:
                break
            last = self.labels
    
    def predict(self, X_new: np.array, n_neighbours: int=3):
        pass
    
    def __initCentroids(self, X: np.array, k: int):
        centroids = np.array([])
        for i in range(k):
            if i == 0:
                np.append(centroids, X[np.random.randint(0, X.shape[0])])
                continue
            probabilities = self.__computeCentroidProb(centroids, X)
            new_centroid = X[np.random.choice(np.arange(0, X.shape[0]), p=probabilities)]
            np.append(centroids, new_centroid)
        self.centroids = centroids
            
    def __distance(self, x, y):
            return ((x - y).T.dot(x - y))
        
    def __distances(self, p: np.array, M: np.array):
        return np.float32([self.distance(p, i) for i in M])
    
    def __computeCentroidProb(self, centroids: np.array, X: np.array):
        min_distances = np.array([])
        for x in X:
            distances = [self.distance(x, i) for i in centroids]
            np.append(min_distances, min(distances))
        return min_distances / min_distances.sum()
    
class KChooser:
    def __init__(self, inertia: np.array, max_n: int=20):
        self.inertia = inertia
        self.max_n = max_n
        
    def __call__(self):
        for i in range(2, self.max_n - 1):
            p = (self.inertia[i] / self.inertia[i + 1]) - (self.inertia[i] / self.inertia[i - 1])
            if 0 <= p <= 1:
                break
        return i + 1
    