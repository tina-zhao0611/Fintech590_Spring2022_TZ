import pandas as pd
import numpy as np
import scipy.stats as st
import math
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import sys,os
from sympy import div
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

from RiskMgmnt import getReturn, getVaR

#read input and calculate the arithmetic return
data = pd.read_csv("Week04\\DailyPrices.csv")

return_mat = getReturn.return_calculate_mat(data)
R_INTC = return_mat["INTC"]

r_mean =  np.mean(R_INTC) #it need to be add back when calculating risk in dollars

print(r_mean)
latest_value = data.loc[data.shape[0] - 1]["INTC"]

#calculate VaR
#compare distribution assumptions with emperical data

# #===========================================================================
VaR_Normal = getVaR.normal(R_INTC, 0.05)


# #===========================================================================
# #Exponentialy weighted Normal

VaR_Normal_w = getVaR.normal_w(R_INTC)



# #===========================================================================
VaR_t_mle = getVaR.t_mle(R_INTC)



# #===========================================================================
# #Historic simulation
VaR_hist = getVaR.hist(R_INTC)

print(VaR_Normal)
print(VaR_Normal_w)
print(VaR_t_mle)
print(VaR_hist)


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