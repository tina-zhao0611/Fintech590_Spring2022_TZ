'''
module for converting matrics into PSD
includes two implemented methods
verified to be working as expected in Week03

func1: near_psd(a, epsilon=0.0):
    uses Rebonato and Jackel’s method
    a is the passed in matrix to be converted
    return: converted PSD matrix
        
func2: def higham_psd(A, max_iterations=1000, tolerance = 1e-9):
    uses Higham's method
    A is the passed in matrix to be converted
    can specify max iterations and tolerance
    return: converted PSD matrix

'''

import numpy as np
from numpy.linalg import eigh

def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD =  np.array([])
    out = a.copy()

    #calculate the correlation matrix if we got a covariance
    if sum(np.diag(out) == 1) != n:
        invSD =np.diag(1 / np.sqrt(np.diag(out)))
        out = np.matmul(invSD, np.matmul(out, invSD))

    #SVD, update the eigen value and scale
    vals, vecs = eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1/(np.matmul((vecs* vecs), vals))
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = np.matmul(np.matmul(T,vecs), l)
    out = np.matmul(B, np.transpose(B))
    #Add back the variance
    if invSD.size != 0 :
        invSD = np.diag(1 / np.diag(invSD))
        out = np.matmul(invSD, np.matmul(out, invSD))
    return out


def getPS(R, weights):
    w_half = np.sqrt(weights)
    R_wtd = w_half * R *w_half
    vals, vecs = eigh(R_wtd)
    #let the eigenvalue system with negative values set to 0
    R_wtd = np.matmul((vecs * np.maximum(vals, 0)), (np.transpose(vecs)))
    X = (1 / w_half) * R_wtd *(1 / w_half)
    return X

def getPU(X):
    #assuming w is diagonal
    Y = X
    np.fill_diagonal(Y, 1)
    return Y

def f_norm (M1, M2, weights):
    w_half = np.sqrt(weights)
    A_w = w_half * (M1 - M2) *w_half
    f_norm = np.sum(A_w * A_w)
    return f_norm
    

def wgtnorm(A, Y, Y_old, weights):

    norm_Y = f_norm(Y,  A, weights)
    norm_Y_old = f_norm(Y_old,  A, weights)
    norml = norm_Y - norm_Y_old
    return norml

def higham_psd(A, max_iterations=1000, tolerance = 1e-9):
    
    n = A.shape[0]
    delta_s = np.zeros(np.shape(A))

    norml = np.inf

    weights = np.ones(np.shape(A)[0])
    Y = np.copy(A)

    iteration = 0
    
    while (norml > tolerance) or (sum(eigh(Y)[0]>-1e-8) < n):
    # if it converge, it is guarenteed to be psd
        if iteration > max_iterations:
            print("No solution found in " + str(max_iterations) + " iterations")
            return A

        R = Y - delta_s
        
        #PS update
        X = getPS(R, weights)

        delta_s = X - R       
        
        Y_old = np.copy(Y)
        #PU update
        Y = getPU(X)
     
        #get norm
        norml = wgtnorm(A, Y, Y_old, weights)
        
        iteration += 1
    return Y
