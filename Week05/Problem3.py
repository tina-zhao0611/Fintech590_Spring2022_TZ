import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np

from RiskMgmnt import getReturn

def get_netValue_series(prices, allocate):
    netValue = pd.DataFrame({"Date": prices["Date"], "value":0})
    T = prices.shape[0]
    for i in range (T):
        current = pd.DataFrame({"Stock": prices.columns, "price":prices.iloc[i,:]}) 
        #merge to get the prices of holding stock and holdings on the same row
        temp = pd.merge(current, allocate, on = "Stock", how="inner")    
        currentValue = sum(temp["price"] * temp["Holding"])
        netValue.loc[i, "value"] = currentValue
    return netValue


prices = pd.read_csv("Week05\\DailyPrices.csv")
holdings = pd.read_csv("Week05\\portfolio.csv")

groups = holdings.groupby("Portfolio")

portfolio_p = []
portfolio_r = []
current_p = []
for portfolio in groups:
    df_p = get_netValue_series(prices, portfolio[1])
    portfolio_p.append(df_p)
    current_p.append(df_p.loc[df_p.shape[0] - 1]["value"])
    df_r = getReturn.return_calculate(df_p, "ARITHMETIC", "value")["return"]
    portfolio_r.append(df_r)
    
all_p = pd.concat(portfolio_p, axis=1)
netValue_all = pd.DataFrame({"Date": prices["Date"], "price":np.sum(all_p.value, axis =1)})
current_p_all = netValue_all.loc[netValue_all.shape[0] - 1]["price"]
netValue_r = getReturn.return_calculate(netValue_all, "ARITHMETIC", "price")["return"]


from RiskMgmnt import getVaR, getES
for i in range(3):
    VaR_t = getVaR.T(portfolio_r[i])
    ES_t = getES.T(portfolio_r[i])
    
    print("\n----Portfolio {:d}----".format(i + 1))
    print("VaR[Return] {:.2%}".format(VaR_t))
    print("VaR[portfolio Value]: (a loss of)  $", -current_p[i] * VaR_t) #the mean should be add back
    print("ES[Return] {:.2%}".format(ES_t))
    print("ES[portfolio Value]: (a loss of)  $", -current_p[i] * ES_t) #the mean should be add back


# 3 portfolios combined
VaR_t = getVaR.T(netValue_r)
ES_t = getES.T(netValue_r)

print("\n----Total----")
print("VaR[Return] {:.2%}".format(VaR_t))
print("VaR[portfolio Value]: (a loss of)  $", -current_p_all * VaR_t) #the mean should be add back
print("ES[Return] {:.2%}".format(ES_t))
print("ES[portfolio Value]: (a loss of)  $", -current_p_all * ES_t) #the mean should be add back
