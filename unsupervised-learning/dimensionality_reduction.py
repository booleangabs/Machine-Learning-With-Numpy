import numpy as np

class PCA:
    def __init__(self, threshold: float, n_components: int=None):
        if threshold == None:
            self.mode = 'number'
            self.n_components = n_components
        else:
            self.mode = 'threshold'    
            self.threshold = threshold
        
    def fit(self, X: np.array):
        n_samples, n_dimensions = X.shape
        self.mean_vector = X.mean(0)
        self.A = X - self.mean_vector
        Sigma = np.cov(self.A, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eig(Sigma)
        indices = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[indices]
        eigen_vectors = eigen_vectors[:, indices]
        
        explained_variance_ratio = eigen_values / eigen_values.sum()
        if self.mode == 'threshold':
            cumulative_expl_var = np.cumsum(explained_variance_ratio)
            self.n_components = np.where(cumulative_expl_var > self.threshold)[0][0]
        self.explained_variance_ratio = explained_variance_ratio[:self.n_components]
        self.eigen_values = eigen_values[:self.n_components]
        self.eigen_vectors = eigen_vectors[:, :self.n_components]
        
    def transform(self) -> np.array:
        return self.A @ self.eigen_vectors
    
    def fit_transform(self, X: np.array) -> np.array:
        self.fit(X)
        return self.transform()
    
    def reconstruct(self, X_hat) -> np.array:
        return (X_hat @ self.eigen_vectors.T) + self.mean_vector
