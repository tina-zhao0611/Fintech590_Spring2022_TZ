import numpy as np
from numpy.linalg import eigh
import time
import matplotlib.pyplot as plt

import sys,os

from sympy import div
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

from RiskMgmnt import getPSD
#=============================================================================
#implementing chol_psd function
#=============================================================================
#generate the same test case from the class
# n = 5
# sigma = np.zeros([n, n]) + 0.9
# for i in range(5):
#     sigma[i, i] = 1.0

# #make the matrix PSD
# sigma[0, 1] = 1.0
# sigma[1, 0] = 1.0

# print("-------------chol_psd--------------")
# print("original isPSD:", getPSD.isPSD(sigma))
# print(sigma)

# new = getPSD.chol_psd(sigma)
# print("new isPSD:", getPSD.isPSD(new))
# print(new)

#=============================================================================
# sigma[0,1] = 0.7357
# sigma[1,0] = 0.7357
# eigh(sigma) #has a value significant below zero
# chol_psd(root,sigma) #this will cause the function to fail
#=============================================================================

#=============================================================================
#implementing the near_psd function
#=============================================================================

n=8
sigma = np.zeros([n, n]) + 0.9
for i in range(n):
    sigma[i, i] = 1.0

#make the matrix non-definite
sigma[0,1] = 0.7357
sigma[1,0] = 0.7357
                
print("-------------near_psd--------------")
print("original isPSD:", getPSD.isPSD(sigma))
print(sigma)

new = getPSD.near_psd(sigma)

print("new isPSD:", getPSD.isPSD(new))
print(new)

#=============================================================================
# implement higham's method
#=============================================================================
n=8
sigma = np.zeros([n, n]) + 0.9
for i in range(n):
    sigma[i, i] = 1.0

#make the matrix non-definite
sigma[0,1] = 0.7357
sigma[1,0] = 0.7357
                
print("-------------near_psd--------------")
print("original isPSD:", getPSD.isPSD(sigma))
print(sigma)

new = getPSD.higham_psd(sigma)

print("new isPSD:", getPSD.isPSD(new))
print(new)

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
    psd_n = getPSD.near_psd(sigma)
    e1 = time.time()
    t_n = e1 - s1
    f_norm_n = np.sum((sigma - psd_n) ** 2)
    
    s2 = time.time()
    psd_higham = getPSD.higham_psd(sigma)
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

    
# plt.cla()
# plt.plot(n, norm_n, label = "nearest_psd")
# plt.plot(n, norm_h, label = "higham_psd")
# plt.legend(loc=1)
# plt.title("Frobenius norm comparison")
# plt.savefig("Week03\\Plots\\Problem2_plot1.png")

# plt.cla()
# plt.plot(n, t_n, label = "nearest_psd")
# plt.plot(n, t_h, label = "higham_psd")
# plt.legend(loc=1)
# plt.title("Runtime comparison")
# plt.savefig("Week03\\Plots\\Problem2_plot2.png")