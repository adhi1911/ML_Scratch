import numpy as np

class myPCA:
    def __init__(self):
        self.k = None
        self.components = None
        self.mean = None
        self.variance_share = None

    def find_k(self, X, threshold = 0.95):
        C = X.T @ X / X.shape[0]
        eigen_values, eigen_vectors = np.linalg.eigh(C)
        eigen_values = np.sort(eigen_values)[::-1]  # Sort and reverse in one step
        total = eigen_values.sum()
        for k in range(X.shape[0]):  # Corrected from x.shape[0] to X.shape[0]
            if eigen_values[:k+1].sum() / total >= threshold:  # Corrected from x to v
                return k + 1
        self.k = len(eigen_values)
        return self.k
    
    def fit(self, x, k = None):
        self.x = x
        if k is not None:
            self.k = k
        else:
            self.k = self.find_k(x)

        '''
        1. centre data
        2. compute covariance matrix
        3. compute eigenvalues and eigenvectors
        4. sort eigenvectors by decreasing eigenvalues
        5. choose k eigenvectors that correspond to k largest eigenvalues
        '''
        # centre data
        self.mean = np.mean(self.x , axis =0)
        x = self.x - self.mean

        # compute covariance matrix
        C = x.T @ x / x.shape[0]
        values, vectors = np.linalg.eigh(C)

        #sort eigenvectors by decreasing eigenvalues
        vectors = vectors[:,::-1]
        values = values[::-1]

        #store the first k eigenvectors
        self.components = vectors[:,:self.k]
        self.eigenvalues = values[:self.k]
        self.variance_share = np.sum(values[:self.k])/np.sum(values)



    def transform(self, x):
        '''
        6. Construct a new matrix with the k eigenvectors
        7. Project the original data onto the new matrix
        '''
        x = x - self.mean
        self.transformed = x @ self.components
        return self.transformed
    
    def reconstruct(self, x):
        '''
        8. Reconstruct the original data from the projected data
        '''
        self.reconstructed = self.transformed @ self.components.T + self.mean
        return self.reconstructed
    
    def reconstruction_error_(self, x):
        '''
        9. Compute the reconstruction error
        '''
        self.error = np.mean(np.linalg.norm(x - self.reconstructed, axis = 1)**2)
        return self.error
    
    def get_eigenvalues(self):
        return self.eigenvalues
    
    def get_eigenvectors(self):
        return self.components