'''
Author: Siyun Wang
Version: 1
'''

import numpy as np

'''
helper functions
'''
def soft_threshOp(X, tau):
    return np.sign(X) * ((np.abs(X) - tau) > 0) * (np.abs(X) - tau)

def sigma_threshOp(X, tau):
    U,S,V = np.linalg.svd(X)
    S_mat = np.zeros_like(X)
    S_mat[:len(S), :len(S)] = np.diag(S)
    return U.dot(soft_threshOp(S_mat, tau).dot(V))


class RPCA():
    
    '''
    perform robust principle component analysis (RPCA) on a given data set (matrix)
    '''
    
    def __init__(self, mu=None, lam=None, tol=1e-6, maxIter=1000, verbose=False):
        '''
        INPUTS:

        tol: positive float, max elementwise difference between the input matrix and the decomposed matrix
             (controls) the number of iterations.
        maxIter: positive integer, hard limit of number of iteration.
        verbose: bool, whether to print the current error.
        '''
        if mu is None:
            self.mu = 10/np.sqrt(max(X.shape))
        else:
            self.mu = mu
            
        if lam is None:
            self.lam = 1/np.sqrt(max(X.shape))
        else:
            self.lam = lam
            
        self.tol = tol
        self.maxIter = maxIter
        self.verbose = verbose
      
    def decompose(self,X):
        '''
        Decomoses the matrix X to a low rank matrix L and a sparse matrix S, X = L + S.
        
        OUTPUTS:
        L,S: low rank and sparse matrices resp., of the same shape as X.
        '''
        L = np.zeros_like(X)
        S = np.zeros_like(X)
        R = np.zeros_like(X)
        
        for i in range(self.maxIter):
            
            L = sigma_threshOp(X-S+(1/self.mu)*R, 1/self.mu)
            S = soft_threshOp(X-L+(1/self.mu)*R, self.lam/self.mu)
            resid = X - L - S
            R += self.mu * resid
            
            err = np.linalg.norm(resid)
            
            if self.verbose and (np.mod(i+1, 50) == 0):
                print("Iteration %d, remaining error: %f" % (i+1, err))
                
            if err < self.tol:
                print("RPCA: Early break, number of iteration: %d." % (i+1))
                break
                
        return L, S
                
