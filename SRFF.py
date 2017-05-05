
import numpy as np # Thinly−wrapped numpy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

class SRFF(BaseEstimator):
    def __init__(self, cdf, a = 2, p = 20, D = 500):
        self.cdf = cdf
        self.a = a
        self.p = p
        #D (number of MonteCarlo samples)
        self.D = D
        self.fitted = False
        
    def random_sample(self):
        R = np.zeros(self.D)
        for i in range(self.D):
            R[i] = max(i for r in [np.random.random()] for i,c in self.cdf if c <= r)
    
        return R.reshape(self.D,1)
    
    def fit(self, X, y=None):
        """ Generates MonteCarlo random samples """
        dim = X.shape[1]
        #Generate D iid samples from p(w) 
        self.w = self.random_sample()
        self.gauss = np.random.randn(self.D,dim)
        w_normalization = np.sqrt((self.gauss**2).sum(axis=1))
        self.gauss /= w_normalization.reshape(-1,1)
        
        self.w = np.repeat(self.w, dim, axis = 1) * self.gauss
        #Generate D iid samples from Uniform(0,2*pi) 
        self.u = 2*np.pi*np.random.rand(self.D)
        self.fitted = True
        return self
    
    def transform(self,X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        if not self.fitted:
            raise NotFittedError("RBF_MonteCarlo must be fitted beform computing the feature map Z")
        #Compute feature map Z(x):
        Z = np.sqrt(2/self.D)*np.cos(X.dot(self.w.T) + self.u[np.newaxis,:])
        return Z
    
    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        if not self.fitted:
            raise NotFittedError("RBF_MonteCarlo must be fitted beform computing the feature map Z")
        Z = self.transform(X)
        K = Z.dot(Z.T)
        return K
    