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
        self.X = X
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
            if np.all(last == self.labels):
                break
            last = self.labels
        self.inertia = self.__getInertia()
        
    def predict(self, X_new: np.array, n_neighbours: int=3):
        classifier = KNN(n_neighbours)
        classifier.fit(self.X, self.labels)
        return classifier.predict(X_new)
    
    def __initCentroids(self, X: np.array, k: int):
        centroids = []
        centroids.append(X[np.random.randint(0, X.shape[0])])
        for _ in range(k - 1):
            probabilities = self.__computeCentroidProb(centroids, X)
            new_centroid = X[np.random.choice(np.arange(0, X.shape[0]), p=probabilities)]
            centroids.append(new_centroid)
        self.centroids = centroids
            
    def __distance(self, x, y) -> float:
            return ((x - y).T.dot(x - y))
        
    def __distances(self, p: np.array, M: np.array) -> np.float32:
        return np.float32([self.__distance(p, i) for i in M])
    
    def __computeCentroidProb(self, centroids: list, X: np.array) -> np.float32:
        min_distances = []
        for x in X:
            distances = [self.__distance(x, i) for i in centroids]
            min_distances.append(min(distances))
        min_distances = np.float32(min_distances)
        return min_distances / min_distances.sum()
    
    def __getInertia(self) -> float:
        classes = {i:self.X[self.labels == i] for i in range(self.n_clusters)}
        inertia = 0
        for i in classes:
            inertia += (self.__distances(self.centroids[i], classes[i])).sum()
        return inertia
    
class KChooser:
    def __init__(self, max_n: int=20):
        self.max_n = max_n
        self.chosen_n = -1
        
    def __call__(self, X: np.array, iterations: int=3):
        ns = []
        inertias = np.zeros((iterations, self.max_n - 1))
        threshold = 1 / (1 + ((iterations / 10) - 3 / 10))
        for i in range(iterations):
            chosen_n, inertias[i] = self.__call(X, threshold)
            ns.append(chosen_n)
        self.average_inertias = inertias.mean(0)
        self.chosen_n = int(np.array(ns).mean()) + 1
            
    def __call(self, X: np.array, threshold: float=1):
        inertias = []
        for n in range(1, self.max_n):
            kmeans = KMeans(n)
            kmeans.fit(X, 3000)
            inertias.append(kmeans.inertia)
        for i in range(1, self.max_n - 2):
            p = (inertias[i] / inertias[i + 1]) - (inertias[i] / inertias[i - 1])
            if 0 < p < threshold:
                break
        return i, np.array(inertias)
    