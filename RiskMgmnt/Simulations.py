'''
module for simulating samples based on covariance matrix

notice: 
    simulated result are made to be centered at 0
    the real mean from original data should be added back if needed

func1: sim_direct(cov_mat, nsim)
    directly draw samples based on the covariance matrix
    (first do Cholesky Factorization to get "L")
    cov_mat is the passed in covariance matrix (N*N)
    nsim is the number of simulation
    return: Matrix of simulated data (nsim * N)
    
func2: sim_pca(cov_mat, nsim, target):  
    uses PCA to reduce the dimentions and simulate
    cov_mat is the passed in covariance matrix (N*N)
    nsim is the number of simulation
    target is the targeting proportion of explained by selected eigen values
    regurn: Matrix of simulated data (nsim * N)

'''

import numpy as np
from numpy.linalg import eigh

#function for doing Cholesky Factorization
def chol_psd(a):
    n = a.shape[0]
    root = np.zeros([n, n])

    # loop over columns
    for j in range(n):
        s = 0
        if j > 0:
            s = np.matmul(root[j, :j], np.transpose( root[j, :j]))

        temp = a[j, j] - s
        #check if around 0
        if -1e-8 <= temp <= 0:
            temp = 0.0

        root[j,j] = np.sqrt(temp)

        if root[j,j] == 0:
            continue #leave them to be zero
            
        ir = 1.0/root[j, j]

        for i in range(j + 1, n):
            s = np.matmul(root[i, :j], np.transpose( root[j, :j]))
            root[i, j] = (a[i, j] - s) * ir

    return root

def sim_direct(cov_mat, nsim):
    n = cov_mat.shape[0]
    root = chol_psd(cov_mat)
    
    #draw samples
    z = np.random.normal(size=(n, nsim)) 
    result = np.matmul(root, z).T
    return result

def sim_pca(cov_mat, nsim, target):    
    eigValue, eigVectors = eigh(np.array(cov_mat,dtype=float))
    
    #keep positive real eigen values
    x = eigValue.shape[0]
    for i in range (x):
        if (eigValue[i] < 1e-8) or (np.imag(eigValue[i]) != 0):
            eigValue[i] = 0
    eigValue = np.real(eigValue)
    
    #calculate the explained variance ratio of each eigen value
    tot = sum(eigValue)
    var_exp = eigValue/tot
    
    #make the vectors the same order as eigValues
    idx = np.argsort(eigValue)[::-1]
    eigVectors = eigVectors[:, idx]
    
    eigValue.sort()
    eigValue = eigValue[::-1] #sort and flip it
    var_exp.sort()
    var_exp = var_exp[::-1] #sort and flip it
    
    cum_var_exp = np.cumsum(var_exp)
    n_c = (cum_var_exp >= target).argmax(axis=0) #number of components needed to hit the target
    eigVectors = eigVectors[:, :n_c+1]
    eigValue = eigValue[:n_c+1]
    
    B = np.matmul(eigVectors, np.diag(np.sqrt(eigValue)))
    
    m = eigValue.shape[0]
    r = np.random.normal(size=(m, nsim)) 
    result = np.matmul(B, r).T
    return result
