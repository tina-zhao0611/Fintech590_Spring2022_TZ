import numpy as np
from numpy.linalg import eigh
import time
import matplotlib.pyplot as plt



#=============================================================================
#implementing chol_psd function
#=============================================================================

def chol_psd(root, a):

    # root = np.full(sigma.shape, 0.0, dtype='float64')
    n = a.shape[0]

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

#generate the same test case from the class
n = 5
sigma = np.zeros([n, n]) + 0.9
for i in range(5):
    sigma[i, i] = 1.0

root = np.zeros([n, n])

#make the matrix PSD
sigma[0, 1] = 1.0
sigma[1, 0] = 1.0

chol_psd(root, sigma)
np.matmul(root, np.transpose(root))

#=============================================================================
# sigma[0,1] = 0.7357
# sigma[1,0] = 0.7357
# eigh(sigma) #has a value significant below zero
# chol_psd(root,sigma) #this will cause the function to fail
#=============================================================================

#=============================================================================
#implementing the near_psd function
#=============================================================================

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


n=500
sigma = np.zeros([n, n]) + 0.9
for i in range(n):
    sigma[i, i] = 1.0

#make the matrix non-definite
sigma[0,1] = 0.7357
sigma[1,0] = 0.7357
                
a = near_psd(sigma)

sum(eigh(sigma)[0]>-1e-8) #original matrix is not psd
sum(eigh(a)[0]>-1e-8) #adjusted matrix is psd

#=============================================================================
# implement higham's method
#=============================================================================

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



#=============================================================================
# comparing resulting matrix and runtime
#=============================================================================
def get_nonpsd(n):
    sigma = np.zeros([n, n]) + 0.9
    for i in range(n):
        sigma[i, i] = 1.0
    sigma[0,1] = 0.7357
    sigma[1,0] = 0.7357
    return sigma

def com_nearest_higham(n):
    sigma = get_nonpsd(n)
    
    s1 = time.time()
    psd_n = near_psd(sigma)
    e1 = time.time()
    t_n = e1 - s1
    f_norm_n = np.sum((sigma - psd_n) ** 2)
    
    s2 = time.time()
    psd_higham = higham_psd(sigma)
    e2 = time.time()
    t_h = e2 - s2
    f_norm_h = np.sum((sigma - psd_higham) ** 2)

    print("n = ", n, "\n       Runtime: nearest_psd: ", t_n, "higham: ", t_h,
          "\nFrobenius Norm: nearest_psd: ", f_norm_n, "higham: ", f_norm_h)
    return [t_n, f_norm_n, t_h, f_norm_h]
    

n = [100, 200, 500, 1000]
t_n = []
norm_n = []
t_h = []
norm_h = []

for i in n:
    a = com_nearest_higham(i)
    t_n.append(a[0])
    norm_n.append(a[1])
    t_h.append(a[2])
    norm_h.append(a[3])

    
plt.cla()
plt.plot(n, norm_n, label = "nearest_psd")
plt.plot(n, norm_h, label = "higham_psd")
plt.legend(loc=1)
plt.title("Frobenius norm comparison")
plt.savefig("Week03\\Plots\\Problem2_plot1.png")

plt.cla()
plt.plot(n, t_n, label = "nearest_psd")
plt.plot(n, t_h, label = "higham_psd")
plt.legend(loc=1)
plt.title("Runtime comparison")
plt.savefig("Week03\\Plots\\Problem2_plot2.png")