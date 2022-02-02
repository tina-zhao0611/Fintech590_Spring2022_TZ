import pandas as pd
import numpy as np
from numpy.linalg import eigh
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time

#calculate the weights vector with n elements, passing in nambda
def expWeight(n, nambda):
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = (1 - nambda) * pow(nambda, n -i + 1) 
    weights = weights / sum(weights)
    return weights

#calculate the weighted variance of one variable
def var_1array(x, w):
    var = 0;
    mean_x = np.mean(x)
    n = len(x)
    for i in range(n):
        var += w[i] * ((x[i] - mean_x)**2)
    return var

#getting variance vector (df is the dataframe holding the data)
def expWeightedVar(df, nambda):
    n, m = df.shape
    weights = expWeight(n, nambda)
    var_v = pd.DataFrame(index = df.columns[1:],columns=[0]) #the first column is the date
    for i in range(m - 1):
        var_v.iloc[i] = var_1array(df.iloc[:, i + 1], weights)
    var_v = pd.Series(var_v[0].values, dtype = float)
    return var_v

#calculate the weighted covariance between two variables
def cov_2arrays(x, y, w):
    cov = 0;
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    n = len(x)
    for i in range(n):
        cov += w[i] * (x[i] - mean_x) * (y[i] - mean_y)
    return cov

#getting correlation matrix (df is the dataframe holding the data)
def expWeightedCorr(df, nambda):
    n, m = df.shape
    weights = expWeight(n, nambda)
    corr = pd.DataFrame(index = df.columns[1:], columns = df.columns[1:]) #the first column is the date
    for i in range(m - 1):
        for j in range(m - 1):
            s1 = np.sqrt(var_1array(df.iloc[:, i + 1], weights))
            s2 = np.sqrt(var_1array(df.iloc[:, j + 1], weights))
            corr.iloc[i][j] = cov_2arrays(df.iloc[:, i + 1], df.iloc[:, j + 1], weights)/(s1 * s2)
    return corr


#reading the input data
data = pd.read_csv("Week03\\DailyReturn.csv")

#Pearson correlation
corr_P = data.corr()
#simple variance
var_P = data.var()
#EW corrolation
corr_EW = expWeightedCorr(data, 0.97)
#EW variance
var_EW = expWeightedVar(data, 0.97)

#get covariance matrix from correlation matrix and variance vactor
def getCOV(df_corr, df_var):
    a=pd.DataFrame(df_var.values.T)
    cov_m = np.matmul(np.sqrt(a),np.sqrt(np.transpose(a)))*(df_corr.values)
    return cov_m
    
#Pearson correlation + simple variance
cov_p_s = getCOV(corr_P, var_P)
#Pearson correlation + EW variance
cov_p_EW = getCOV(corr_P, var_EW)
#EW correlation + simple variance
cov_EW_s = getCOV(corr_EW, var_P)
#EW correlation + EW variance
cov_EW_EW = getCOV(corr_EW, var_EW)

#implemented decomposition function from problem2
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

def sim_direct(cov_mat, nsim):
    n = cov_mat.shape[0]
    root = np.zeros([n, n])
    #use the function to get L
    chol_psd(root, cov_mat)
    
    #draw samples
    z = np.random.normal(size=(n, nsim)) 
    s = np.matmul(root, z)
    #excluded the mean as it won't affect the covariance
    return s


#target is the targeted explained ratio
def sim_pca(mat_cov, nsim, target):    
    eigValue, eigVectors = eigh(np.array(mat_cov,dtype=float))
    
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
    s = np.matmul(B, r)
    return s

# =============================================================================
# Pearson correlation + simple variance
# =============================================================================
# PCA 50% explained
fnorm1=[]
runtime1=[]

start = time.time()

p_s_50 = sim_pca(cov_p_s, 25000, 0.5)

end = time.time()
distance = np.cov(p_s_50)- cov_p_s
f_norm = norm(distance, 'fro')
fnorm1.append(f_norm)
runtime1.append(end-start)
print("--------Pearson correlation + simple variance-------")
print ("distance from real covariance(50% explained): ", f_norm)
print ("runtime (50% explained):", str(end-start))

# PCA 75% explained
start = time.time()

p_s_75 = sim_pca(cov_p_s, 25000, 0.75)

end = time.time()
distance = np.cov(p_s_75)- cov_p_s
f_norm = norm(distance, 'fro')
fnorm1.append(f_norm)
runtime1.append(end - start)
print ("distance from real covariance(75% explained): ", f_norm)
print ("runtime (75% explained):", str(end - start))

# PCA 100% explained
start = time.time()

p_s_100 = sim_pca(cov_p_s, 25000, 0.99999)
# when using "1" in the function, it returns with a runtime smaller then lower accuracy
# and a larger Frobenius norm, haven't fund the cause yet
end = time.time()
distance = np.cov(p_s_100)- cov_p_s
f_norm = norm(distance, 'fro')
fnorm1.append(f_norm)
runtime1.append(end-start)
print ("distance from real covariance(100% explained): ", f_norm)
print ("runtime (100% explained):", str(end-start))

start = time.time()

p_s = sim_direct(cov_p_s.values, 25000)

end = time.time()
distance = np.cov(p_s)- cov_p_s
f_norm = norm(distance, 'fro')
fnorm1.append(f_norm)
runtime1.append(end-start)
print ("distance from real covariance(direct): ", f_norm)
print ("runtime (direct):", str(end-start))


x_axis = ["PCA 50%", "PCA 75%", "PCA 100%", "Direct"]
plt.cla()
fig,ax1 = plt.subplots()
ax1.plot(x_axis, fnorm1, label = "Frobenius Norm", color='red')
plt.legend(loc=2)
ax2 = ax1.twinx() 
ax2.plot(x_axis, runtime1, label = "Runtime")
plt.legend(loc=1)
plt.title("Pearson correlation + simple variance")
fig.tight_layout()
plt.savefig("Week03\\Plots\\Problem3_plot1.png")

# =============================================================================
# Pearson correlation + EW variance
# =============================================================================
# PCA 50% explained
fnorm2=[]
runtime2=[]

start = time.time()

start = time.time()

p_EW_50 = sim_pca(cov_p_EW, 25000, 0.5)

end = time.time()
distance = np.cov(p_EW_50)- cov_p_EW
f_norm = norm(distance, 'fro')
fnorm2.append(f_norm)
runtime2.append(end-start)
print("--------Pearson correlation + EW variance-------")
print ("distance from real covariance(50% explained): ", f_norm)
print ("runtime (50% explained):", str(end-start))

# PCA 75% explained
start = time.time()

p_EW_75 = sim_pca(cov_p_EW, 25000, 0.75)

end = time.time()
distance = np.cov(p_EW_75)- cov_p_EW
f_norm = norm(distance, 'fro')
fnorm2.append(f_norm)
runtime2.append(end-start)
print ("distance from real covariance(75% explained): ", f_norm)
print ("runtime (75% explained):", str(end-start))

# PCA 100% explained
start = time.time()

p_EW_100 = sim_pca(cov_p_EW, 25000, 1)

end = time.time()
distance = np.cov(p_EW_100)- cov_p_EW
f_norm = norm(distance, 'fro')
fnorm2.append(f_norm)
runtime2.append(end-start)
print ("distance from real covariance(100% explained): ", f_norm)
print ("runtime (100% explained):", str(end-start))

start = time.time()

p_EW = sim_direct(cov_p_EW.values, 25000)

end = time.time()
distance = np.cov(p_EW)- cov_p_EW
f_norm = norm(distance, 'fro')
fnorm2.append(f_norm)
runtime2.append(end-start)
print ("distance from real covariance(direct): ", f_norm)
print ("runtime (direct):", str(end-start))

x_axis = ["PCA 50%", "PCA 75%", "PCA 100%", "Direct"]
plt.cla()
fig,ax1 = plt.subplots()
ax1.plot(x_axis, fnorm2, label = "Frobenius Norm", color='red')
plt.legend(loc=2)
ax2 = ax1.twinx() 
ax2.plot(x_axis, runtime2, label = "Runtime")
plt.legend(loc=1)
plt.title("Pearson correlation + EW variance")
fig.tight_layout()
plt.savefig("Week03\\Plots\\Problem3_plot2.png")

# =============================================================================
# EW correlation + simple variance
# =============================================================================
# PCA 50% explained
fnorm3=[]
runtime3=[]

start = time.time()

EW_s_50 = sim_pca(cov_EW_s, 25000, 0.5)

end = time.time()
distance = np.cov(EW_s_50)- cov_EW_s
f_norm = norm(distance, 'fro')
fnorm3.append(f_norm)
runtime3.append(end-start)
print("--------EW correlation + simple variance-------")
print ("distance from real covariance(50% explained): ", f_norm)
print ("runtime (50% explained):", str(end-start))

# PCA 75% explained
start = time.time()

EW_s_75 = sim_pca(cov_EW_s, 25000, 0.75)

end = time.time()
distance = np.cov(EW_s_75)- cov_EW_s
f_norm = norm(distance, 'fro')
fnorm3.append(f_norm)
runtime3.append(end-start)
print ("distance from real covariance(75% explained): ", f_norm)
print ("runtime (75% explained):", str(end-start))

# PCA 100% explained
start = time.time()

EW_s_100 = sim_pca(cov_EW_s, 25000, 1)

end = time.time()
distance = np.cov(EW_s_100)- cov_EW_s
f_norm = norm(distance, 'fro')
fnorm3.append(f_norm)
runtime3.append(end-start)
print ("distance from real covariance(100% explained): ", f_norm)
print ("runtime (100% explained):", str(end-start))

start = time.time()

EW_s = sim_direct(cov_EW_s.values, 25000)

end = time.time()
distance = np.cov(EW_s)- cov_EW_s
f_norm = norm(distance, 'fro')
fnorm3.append(f_norm)
runtime3.append(end-start)
print ("distance from real covariance(direct): ", f_norm)
print ("runtime (direct):", str(end-start))

x_axis = ["PCA 50%", "PCA 75%", "PCA 100%", "Direct"]
plt.cla()
fig,ax1 = plt.subplots()
ax1.plot(x_axis, fnorm3, label = "Frobenius Norm", color='red')
plt.legend(loc=2)
ax2 = ax1.twinx() 
ax2.plot(x_axis, runtime3, label = "Runtime")
plt.legend(loc=1)
plt.title("EW correlation + simple variance")
fig.tight_layout()
plt.savefig("Week03\\Plots\\Problem3_plot3.png")

# =============================================================================
# EW correlation + EW variance
# =============================================================================
# PCA 50% explained
fnorm4=[]
runtime4=[]

start = time.time()

EW_EW_50 = sim_pca(cov_EW_EW, 25000, 0.5)

end = time.time()
distance = np.cov(EW_EW_50)- cov_EW_EW
f_norm = norm(distance, 'fro')
fnorm4.append(f_norm)
runtime4.append(end-start)
print("--------EW correlation + EW variance-------")
print ("distance from real covariance(50% explained): ", f_norm)
print ("runtime (50% explained):", str(end-start))

# PCA 75% explained
start = time.time()
EW_EW_75 = sim_pca(cov_EW_EW, 25000, 0.75)

end = time.time()
distance = np.cov(EW_EW_75)- cov_EW_EW
f_norm = norm(distance, 'fro')
fnorm4.append(f_norm)
runtime4.append(end-start)
print ("distance from real covariance(75% explained): ", f_norm)
print ("runtime (75% explained):", str(end-start))

# PCA 100% explained
start = time.time()

EW_EW_100 = sim_pca(cov_EW_EW, 25000, 1)

end = time.time()
distance = np.cov(EW_EW_100)- cov_EW_EW
f_norm = norm(distance, 'fro')
fnorm4.append(f_norm)
runtime4.append(end-start)
print ("distance from real covariance(100% explained): ", f_norm)
print ("runtime (100% explained):", str(end-start))

start = time.time()

EW_EW = sim_direct(cov_EW_EW.values, 25000)

end = time.time()
distance = np.cov(EW_EW)- cov_EW_EW
f_norm = norm(distance, 'fro')
fnorm4.append(f_norm)
runtime4.append(end-start)
print ("distance from real covariance(direct): ", f_norm)
print ("runtime (direct):", str(end-start))

x_axis = ["PCA 50%", "PCA 75%", "PCA 100%", "Direct"]
plt.cla()
fig,ax1 = plt.subplots()
ax1.plot(x_axis, fnorm4, label = "Frobenius Norm", color='red')
plt.legend(loc=2)
ax2 = ax1.twinx() 
ax2.plot(x_axis, runtime4, label = "Runtime")
plt.legend(loc=1)
plt.title("EW correlation + EW variance")
fig.tight_layout()
plt.savefig("Week03\\Plots\\Problem3_plot4.png")