import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore") 
# there's a warning initiated by the optimization which seams to be caused by the version of the package

prices = pd.read_csv("DailyPrices.csv")
holdings = pd.read_csv("portfolio.csv")

groups = holdings.groupby("Portfolio")

#function from problem2
def return_calculate(price, method = "BM", column_name = "price"):
    T = price.shape[0]
    price["p1p0"] = 0

    for i in range(1, T):
        #Classical Brownian Motion
        if(method == "BM"):
            price.loc[i, "p1p0"] = price.loc[i,column_name]  - price.loc[i - 1,column_name] 

        #If other two methods
        else:
            price.loc[i, "p1p0"] = price.loc[i,column_name]  / price.loc[i - 1,column_name] 

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

def get_netValue_series(allocate):
    netValue = pd.DataFrame({"Date": prices["Date"], "value":0})
    T = prices.shape[0]
    for i in range (T):
        current = pd.DataFrame({"Stock": prices.columns, "price":prices.iloc[i,:]}) 
        #merge to get the prices of holding stock and holdings on the same row
        temp = pd.merge(current, allocate, on = "Stock", how="inner")    
        currentValue = sum(temp["price"] * temp["Holding"])
        netValue.loc[i, "value"] = currentValue
    return netValue

portfolio_p = []
portfolio_r = []
current_p = []
r_mean = []
for portfolio in groups:
    df_p = get_netValue_series(portfolio[1])
    portfolio_p.append(df_p)
    current_p.append(df_p.loc[df_p.shape[0] - 1]["value"])
    df_r = return_calculate(df_p, "ARITHMETIC", "value").loc[:,"return"]
    df_r = df_r - np.mean(df_r) #make the mean 0
    r_mean.append(np.mean(df_r))
    portfolio_r.append(df_r)
    
all_p = pd.concat(portfolio_p, axis=1)
netValue_all = pd.DataFrame({"Date": prices["Date"], "price":np.sum(all_p.value, axis =1)})
current_p_all = netValue_all.loc[netValue_all.shape[0] - 1]["price"]

netValue_r = return_calculate(netValue_all, "ARITHMETIC", "price").loc[:,"return"]
netValue_r = netValue_r - np.mean(netValue_r) #make the mean 0
mean_all = np.mean(netValue_r)

#===========================================================================
# select the MLE fitted T distribution method
#functions from problem 2
def likelyhood_t(parameter, x):
    n = parameter[0]
    std = parameter[1]
    
    L = np.sum(st.t.logpdf(x, df = n, loc = 0, scale = std)) #L is the log likelihood
    return -L

def getVaR_t_mle(r, alpha = 0.05):
    cons = ({'type': 'ineq', 'fun': lambda x: x[1] - 0}) #standard deviation should be larger than 0
    mle_model = minimize(likelyhood_t, [r.size, 1], args = r, constraints = cons) #minimizing -L means maximizing L
    return mle_model.x


alpha = 0.05
#each portfolio
for i in range(3):
    mle_estimates = getVaR_t_mle(portfolio_r[i])
    VaR_t_mle = -st.t.ppf(alpha, df = mle_estimates[0], loc = 0, scale = mle_estimates[1])

    print("\n----Portfolio {:d}----".format(i + 1))
    print("VaR[Return] {:.2%}".format(VaR_t_mle))
    print("VaR[portfolio Value]: (a loss of)  $", current_p[i] * (VaR_t_mle - r_mean[i])) #the mean should be add back


# 3 portfolios combined
mle_estimates = getVaR_t_mle(netValue_r)
VaR_t_mle_all = -st.t.ppf(alpha, df = mle_estimates[0], loc = 0, scale = mle_estimates[1])

print("\n----Total----")
print("VaR[Return] {:.2%}".format(VaR_t_mle_all))
print("VaR[portfolio Value]: (a loss of)  $", current_p_all * (VaR_t_mle_all - mean_all)) #the mean should be add back
