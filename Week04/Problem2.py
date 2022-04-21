from tkinter.tix import DirSelectBox
import pandas as pd
import numpy as np
import scipy.stats as st
import math
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# implementing return_calculate function
def return_calculate(price, method = "BM", column_name = "price"):
    T = price.shape[0]
    price["p1p0"] = 0

    for i in range(1, T):
        #Classical Brownian Motion
        if(method == "BM"):
            price.loc[i, "p1p0"] = price.loc[i, column_name]  - price.loc[i - 1, column_name] 

        #If other two methods
        else:
            price.loc[i, "p1p0"] = price.loc[i, column_name]  / price.loc[i - 1, column_name] 

    df_r = price[1:].copy()

    #Classical Brownian Motion
    if(method == "BM"):
        df_r.loc[:, "return"] = df_r.loc[:, "p1p0"]

    #Arithmetic Return
    if(method == "ARITHMETIC"):
        df_r.loc[:, "return"] = df_r.loc[:, "p1p0"] - 1

    #Geometric Brownian Motion
    if(method == "GBM"):
        df_r.loc[:, "return"] = np.log(df_r.loc[:, "p1p0"])

    df_r = df_r.drop(columns="p1p0")
    
    return df_r

#read input and calculate the arithmetic return
data = pd.read_csv("Week04\\DailyPrices.csv")
P_INTC = pd.DataFrame({"Date":data["Date"], "price": data["INTC"]})
R_INTC = return_calculate(P_INTC , "ARITHMETIC").loc[:,"return"]

r_mean =  np.mean(R_INTC) #it need to be add back when calculating risk in dollars
R_INTC = R_INTC - np.mean(R_INTC) #make the mean 0

latest_value = P_INTC.loc[P_INTC.shape[0] - 1]["price"]

print(r_mean)
#calculate VaR
#compare distribution assumptions with emperical data


# #===========================================================================
# #Normal distribution
def getVaR_normal(r, alpha = 0.05):
    u = 0   #already centered, mean = 0
    sigma = np.std(r)
    VaR_Normal = -st.norm.ppf(alpha, loc = u, scale = sigma) 
    return VaR_Normal 

alpha = 0.05
VaR_Normal = getVaR_normal(R_INTC, 0.05)

u = 0   #already centered, mean = 0
sigma = np.std(R_INTC)
# plt.cla()
# R_INTC.hist(bins=100,alpha=0.3,color='k',density=True)  
# R_INTC.plot(kind='kde',style='k--', label = "empirical")
# x = np.linspace(u - 3*sigma, u + 3*sigma) 
# y = st.norm.pdf(x, loc = u, scale = sigma)
# plt.plot(x, y, "r--", linewidth=2, label = "normal distribution")
# plt.axvline(-VaR_Normal, color='b', label='VaR_Normal')
# plt.legend()
# plt.title("distribution comparision")
# plt.savefig("plots\\Problem1_norm.png")


# #===========================================================================
# #Exponentialy weighted Normal

#two functions from last week's project
def expWeight(n, Lambda):
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = (1 - Lambda) * pow(Lambda, n -i + 1) 
    #in the input data, the latest value is at the bottom
    #so generating weights accordingly (largest weight should be at the back)
    weights = weights / sum(weights)
    return weights
def cov_2arrays(x, y, w):
    cov = 0;
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    n = len(x)
    for i in range(n):
        cov += w[i] * (x[i] - mean_x) * (y[i] - mean_y)
    return cov


def getVaR_normal_w(r, alpha = 0.05, Lambda = 0.97):
    
    n = r.size
    w = expWeight(n, Lambda)
    sigma_w = np.sqrt(cov_2arrays(r.values, r.values, w))
    VaR_Normal_w = -st.norm.ppf(alpha, loc = 0, scale = sigma_w) 
    return VaR_Normal_w

Lambda = 0.97
VaR_Normal_w = getVaR_normal_w(R_INTC)

n = R_INTC.size
w = expWeight(n, Lambda)
sigma_w = np.sqrt(cov_2arrays(R_INTC.values, R_INTC.values, w))
# plt.cla()
# R_INTC.hist(bins=100,alpha=0.3,color='k',density=True)  
# R_INTC.plot(kind='kde',style='k--', label = "empirical")
# x = np.linspace(u - 3*sigma_w, u + 3*sigma_w) 
# y = st.norm.pdf(x, loc = 0, scale = sigma_w)
# plt.plot(x, y, "r--", linewidth=2, label = "normal distribution (Exponentially weighted)")
# plt.axvline(-VaR_Normal_w, color='b', label='VaR_Normal')
# plt.legend()
# plt.title("distribution comparision")
# plt.savefig("plots\\Problem1_norm_weighted.png")


# #===========================================================================
# #MLE fitted T distribution
def likelyhood_t(parameter, x):
    n = parameter[0]
    std = parameter[1]
    
    L = np.sum(st.t.logpdf(x, df = n, loc = 0, scale = std)) #L is the log likelihood
    return -L

def getVaR_t_mle(r, alpha = 0.05):
    cons = ({'type': 'ineq', 'fun': lambda x: x[1] - 0}) #standard deviation should be larger than 0
    mle_model = minimize(likelyhood_t, [r.size, 1], args = r, constraints = cons) #minimizing -L means maximizing L
    return mle_model.x

mle_estimates = getVaR_t_mle(R_INTC)
VaR_t_mle = -st.t.ppf(alpha, df = mle_estimates[0], loc = 0, scale = mle_estimates[1])

# plt.cla()
# R_INTC.hist(bins=100,alpha=0.3,color='k',density=True)  
# R_INTC.plot(kind='kde',style='k--', label = "empirical")
# x = np.linspace(u - 3*sigma_w, u + 3*sigma_w) 
# y = st.t.pdf(x, df = mle_estimates[0], loc = 0, scale = mle_estimates[1])
# plt.plot(x, y, "r--", linewidth=2, label = "t distribution (MLE)")
# plt.axvline(-VaR_t_mle, color='b', label='VaR_t_MLE')
# plt.legend()
# plt.title("distribution comparision")
# plt.savefig("plots\\Problem1_t_MLE.png")

# #===========================================================================
#Historic simulation
times = 48
distribution = R_INTC.sample(times, replace=True)
VaR_hist = -np.percentile(distribution, alpha)

print(VaR_Normal)
print(VaR_Normal_w)
print(VaR_t_mle)
print(VaR_hist)
# plt.cla()
# R_INTC.hist(bins=100,alpha=0.3,color='k',density=True)  
# R_INTC.plot(kind='kde',style='k--', label = "empirical")
# distribution.hist(bins=100,alpha=0.3,color='r',density=True) 
# distribution.plot(kind='kde',style='r--',label = "historic simulation")
# plt.axvline(-VaR_hist, color='b', label='VaR_hist')
# plt.legend()
# plt.title("distribution comparision")
# plt.savefig("plots\\Problem1_historic.png")

# print("VaR calculations:",
#       "\n   Normal Distribution: {:.2%}".format(VaR_Normal), "in dollars: {:.3f}".format((VaR_Normal - r_mean) * latest_value))
# print("Exponentially Weighted: {:.2%}".format(VaR_Normal_w), "in dollars: {:.3f}".format((VaR_Normal_w - r_mean) * latest_value))
# print("          MLE fitted T: {:.2%}".format(VaR_t_mle), "in dollars: {:.3f}".format((VaR_t_mle - r_mean) * latest_value))
# print("   Historic Simulation: {:.2%}".format(VaR_hist), "in dollars: {:.3f}".format((VaR_hist - r_mean) * latest_value))

# #===========================================================================
# #out of sample
# #===========================================================================

# df = pd.read_csv("INTC.csv")
# P = pd.DataFrame({"Date":df["Date"], "price": df["Adj Close"]})
# R = return_calculate(P, "ARITHMETIC").loc[:,"return"]
# R = R - np.mean(R)

# plt.cla()
# R.hist(bins=100,alpha=0.3,color='k',density=True)  
# R.plot(kind='kde',style='k--', label = "empirical")
# x = np.linspace(u - 3*sigma, u + 3*sigma) 
# y = st.norm.pdf(x, loc = u, scale = sigma)
# plt.plot(x, y, "r--", linewidth=2, label = "normal distribution")
# plt.axvline(-VaR_Normal, color='b', label='VaR_Normal')
# plt.axvline(np.quantile(R, alpha), color='g', label='real 0.05 quantile')
# plt.legend()
# plt.title("distribution comparision(out of sample)")
# plt.savefig("plots\\Problem1_norm2.png")

# plt.cla()
# R.hist(bins=100,alpha=0.3,color='k',density=True)  
# R.plot(kind='kde',style='k--', label = "empirical")
# x = np.linspace(u - 3*sigma_w, u + 3*sigma_w) 
# y = st.norm.pdf(x, loc = u, scale = sigma_w)
# plt.plot(x, y, "r--", linewidth=2, label = "normal distribution (Exponentially weighted)")
# plt.axvline(-VaR_Normal_w, color='b', label='VaR_Normal')
# plt.axvline(np.quantile(R, alpha), color='g', label='real 0.05 quantile')
# plt.legend()
# plt.title("distribution comparision(out of sample)")
# plt.savefig("plots\\Problem1_norm_weighted2.png")

# plt.cla()
# R.hist(bins=100,alpha=0.3,color='k',density=True)  
# R.plot(kind='kde',style='k--', label = "empirical")
# x = np.linspace(u - 3*sigma_w, u + 3*sigma_w) 
# y = st.t.pdf(x, df = mle_estimates[0], loc = 0, scale = mle_estimates[1])
# plt.plot(x, y, "r--", linewidth=2, label = "t distribution (MLE)")
# plt.axvline(-VaR_t_mle, color='b', label='VaR_t_MLE')
# plt.axvline(np.quantile(R, alpha), color='g', label='real 0.05 quantile')
# plt.legend()
# plt.title("distribution comparision(out of sample)")
# plt.savefig("plots\\Problem1_t_MLE2.png")

# plt.cla()
# R.hist(bins=100,alpha=0.3,color='k',density=True)  
# R.plot(kind='kde',style='k--', label = "empirical")
# distribution.hist(bins=100,alpha=0.3,color='r',density=True) 
# distribution.plot(kind='kde',style='r--',label = "historic simulation")
# plt.axvline(-VaR_hist, color='b', label='VaR_hist')
# plt.axvline(np.quantile(R, alpha), color='g', label='real 0.05 quantile')
# plt.legend()
# plt.title("distribution comparision(out of sample)")
# plt.savefig("plots\\Problem1_historic2.png")