import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)
from RiskMgmnt import getReturn, T_dist_fitter, covEstimate

data = pd.read_csv("Week05\\problem1.csv")

#======================================================
#Fitting normal distribution
#======================================================
def getVaR_ES_normal(data, alpha = 0.05):
    mu = data.mean().values
    data -= mu
    sigma = np.std(data).values

    VaR = np.percentile(data, alpha) + mu
    VaR_Normal = st.norm.ppf(alpha, loc=mu, scale=sigma)
    
    temp = data[data <= VaR].dropna()
    ES = temp.mean().values + mu
    ES_Normal = -mu + sigma * (st.norm.pdf(st.norm.ppf(alpha)))/alpha
    normal_mu = mu
    normal_sigma = sigma
    
    return normal_mu, normal_sigma, VaR, VaR_Normal, ES, -ES_Normal

normal_mu, normal_sigma, VaR, VaR_Normal, ES, ES_Normal = getVaR_ES_normal(data.copy()) 

#======================================================
#Fitting T distribution
#======================================================
def getVaR_ES_T(data, alpha = 0.05):
    # t_m = data.mean().values
    t_df, t_m, t_s = T_dist_fitter.getT_MLE(data)
    # simulate t distribution with estimated parameters
    t = st.t.rvs(df = t_df, loc = t_m, scale = t_s, size = 100000)
    tsim = pd.DataFrame({"tsim": t})
    
    VaR_T = st.t.ppf(alpha, df = t_df, loc = t_m, scale = t_s)
    # np.percentile(tsim, alpha)
    temp = tsim[tsim <= VaR_T].dropna()
    ES_T = temp.mean().values
    return t_df, t_m, t_s, VaR_T, ES_T

t_df, t_m, t_s, VaR_T, ES_T = getVaR_ES_T(data.copy())

print("VAR_NORMAL (5%): ", VaR_Normal, "ES_NORMAL: ", ES_Normal)
print("VAR_T (5%): ", VaR_T, "ES_T: ", ES_T)

#plotting VaR comparison
plt.cla()

data.x.hist(bins=100,alpha=0.3,color='k',density=True)  
data.x.plot(kind='kde',style='k--', label = "data")

x = np.linspace(normal_mu - 3 * normal_sigma, normal_mu + 3 * normal_sigma) 
y_normal = st.norm.pdf(x, loc = normal_mu, scale = normal_sigma)
y_t = st.t.pdf(x, df = t_df, loc = t_m, scale = t_s)

plt.plot(x, y_normal, "r--", linewidth=2, label = "normal distribution")
plt.plot(x, y_t, "b--", linewidth=2, label = "T distribution")

plt.axvline(VaR, color='g', ls =":", label='real VaR')
plt.axvline(VaR_Normal, color='r', label='VaR_Normal')
plt.axvline(VaR_T, color='b', label='VaR_T')

plt.legend()
plt.title("Model results: VaR estimates (5%)")
plt.savefig("Week05\\plots\\Problem1_VaR.png")


#plotting ES comparison
plt.cla()

data.x.hist(bins=100,alpha=0.3,color='k',density=True)  
data.x.plot(kind='kde',style='k--', label = "data")

x = np.linspace(normal_mu - 3 * normal_sigma, normal_mu + 3 * normal_sigma) 
y_normal = st.norm.pdf(x, loc = normal_mu, scale = normal_sigma)
y_t = st.t.pdf(x, df = t_df, loc = t_m, scale = t_s)

plt.plot(x, y_normal, "r--", linewidth=2, label = "normal distribution")
plt.plot(x, y_t, "b--", linewidth=2, label = "T distribution")

plt.axvline(ES, color='g', ls =":", label='real ES')
plt.axvline(ES_Normal, color='r', label='ES_Normal')
plt.axvline(ES_T, color='b', label='ES_T')

plt.legend()
plt.title("Model results: ES estimates (5%)")
plt.savefig("Week05\\plots\\Problem1_ES.png")

normal_mu, normal_sigma, VaR, VaR_Normal, ES, ES_Normal = getVaR_ES_normal(data.copy(), 0.01) 
t_df, t_m, t_s, VaR_T, ES_T = getVaR_ES_T(data.copy(), 0.01)
#plotting VaR comparison
plt.cla()

data.x.hist(bins=100,alpha=0.3,color='k',density=True)  
data.x.plot(kind='kde',style='k--', label = "data")

x = np.linspace(normal_mu - 3 * normal_sigma, normal_mu + 3 * normal_sigma) 
y_normal = st.norm.pdf(x, loc = normal_mu, scale = normal_sigma)
y_t = st.t.pdf(x, df = t_df, loc = t_m, scale = t_s)

plt.plot(x, y_normal, "r--", linewidth=2, label = "normal distribution")
plt.plot(x, y_t, "b--", linewidth=2, label = "T distribution")

plt.axvline(VaR, color='g', ls =":", label='real VaR')
plt.axvline(VaR_Normal, color='r', label='VaR_Normal')
plt.axvline(VaR_T, color='b', label='VaR_T')

plt.legend()
plt.title("Model results: VaR estimates (1%)")
plt.savefig("Week05\\plots2\\Problem1_VaR.png")


#plotting ES comparison
plt.cla()

data.x.hist(bins=100,alpha=0.3,color='k',density=True)  
data.x.plot(kind='kde',style='k--', label = "data")

x = np.linspace(normal_mu - 3 * normal_sigma, normal_mu + 3 * normal_sigma) 
y_normal = st.norm.pdf(x, loc = normal_mu, scale = normal_sigma)
y_t = st.t.pdf(x, df = t_df, loc = t_m, scale = t_s)

plt.plot(x, y_normal, "r--", linewidth=2, label = "normal distribution")
plt.plot(x, y_t, "b--", linewidth=2, label = "T distribution")

plt.axvline(ES, color='g', ls =":", label='real ES')
plt.axvline(ES_Normal, color='r', label='ES_Normal')
plt.axvline(ES_T, color='b', label='ES_T')

plt.legend()
plt.title("Model results: ES estimates (1%)")
plt.savefig("Week05\\plots2\\Problem1_ES.png")

print("VAR_NORMAL (1%): ", VaR_Normal, "ES_NORMAL: ", ES_Normal)
print("VAR_T (1%): ", VaR_T, "ES_T: ", ES_T)