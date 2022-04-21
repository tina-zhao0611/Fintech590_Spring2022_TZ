'''
module for calculating VaR

6 methods are implemented

fitting normal distribution
fitting normal with exponetially weighted covariance
fitting t distribution with MLE
historical simulation
fitting generalized t distribution
get VaR directly from series of data

return the value of VaR (negative value means a loss)

'''
import pandas as pd
import numpy as np
import scipy.stats as st
import math
from scipy.optimize import minimize

#Normal distribution
def normal(data, alpha = 0.05):
    mu = data.mean()
    data -= mu
    sigma = np.std(data)

    VaR_Normal = st.norm.ppf(alpha, loc=mu, scale=sigma)
    return VaR_Normal 


#Normal distribution with exponentially weighted covariance
from RiskMgmnt import expWeighted

def normal_w(data, alpha = 0.05, Lambda = 0.97):
    mu = data.mean()
    sigma_w = expWeighted.cov_w(data, Lambda)

    VaR_Normal_w = st.norm.ppf(alpha, loc = mu, scale = sigma_w) 
    return VaR_Normal_w

#MLE fitted T distribution
def likelyhood_t(parameter, x):
    n = parameter[0]
    std = parameter[1]
    
    L = np.sum(st.t.logpdf(x, df = n, loc = 0, scale = std)) #L is the log likelihood
    return -L

def t_mle(data, alpha = 0.05):
    cons = ({'type': 'ineq', 'fun': lambda x: x[1] - 0}) #standard deviation should be larger than 0
    mle_model = minimize(likelyhood_t, [data.size, 1], args = data, constraints = cons) #minimizing -L means maximizing L
    mle_estimates = mle_model.x
    VaR_t_mle = st.t.ppf(alpha, df = mle_estimates[0], loc = 0, scale = mle_estimates[1])    
    return VaR_t_mle

#Historic simulation
def hist(data, alpha = 0.05, times = 0):
    if times == 0:
        times = round(data.size * 0.8)
    distribution = data.sample(times, replace=True)
    VaR_hist = np.percentile(distribution, alpha)
    return VaR_hist

#generalized t distribution
def T(data, alpha = 0.05):
    t_df, t_m, t_s = st.t.fit(data)
    # simulate t distribution with estimated parameters
    t = st.t.rvs(df = t_df, loc = t_m, scale = t_s, size = 10000)
    tsim = pd.DataFrame({"tsim": t})
    
    VaR_T = np.percentile(tsim, alpha)

    return VaR_T

def empirical(data, alpha = 0.05):
    VaR_emp = np.quantile(data, alpha)
    
    return VaR_emp
