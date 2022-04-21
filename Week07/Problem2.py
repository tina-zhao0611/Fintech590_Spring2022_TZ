import sys,os

# __file__ = "Problem2.py"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import scipy.stats as st
import time

import matplotlib.pyplot as plt

from RiskMgmnt import getVaR, getES, Options, americanBT

# OPTION INFORMATION
current_price = 164.85
rf = 0.0025
r_benefit = 0 # paying discret dividend
current_date = "02/25/2022"

div = [1]
payment_date = "03/15/2022"
multi = 5

# function for calculating porgtolio delta
def getPortfolioDelta(portfolio, currentUnderlyingValue, current_date, div):
    delta = 0
    for row in portfolio.itertuples(name = "options"):
        if (getattr(row, "Type") == "Stock"):
            delta += getattr(row, "Holding")
        if (getattr(row, "Type") == "Option"):
            opt = Options.option(type = getattr(row, "OptionType"), 
                        exp_date = getattr(row, "ExpirationDate"), 
                        K = getattr(row, "Strike"),
                        S0 = currentUnderlyingValue,
                        r_benefit = r_benefit)
            steps = americanBT.getDivT(current_date, getattr(row, "ExpirationDate"), payment_date)[1] * multi
            divT = [americanBT.getDivT(current_date, getattr(row, "ExpirationDate"), payment_date)[0] * multi]

            impVol = getattr(row, "impVol")
            delta += americanBT.Delta(opt, current_date, impVol, rf, steps, div, divT) * getattr(row, "Holding")
           
    return delta

# FUNCTION FOR CALCULATING PORTFOLIO PNL
# def getPortfolioPnL(portfolio, forwardDate, currentUnderlyingValue, underlyingValue, div):
#     print(portfolio)
#     PnL = 0
#     for row in portfolio.itertuples(name = "options"):
#         if (getattr(row, "Type") == "Stock"):
#             PnL += (underlyingValue - currentUnderlyingValue) * getattr(row, "Holding")
#         if (getattr(row, "Type") == "Option"):
#             opt = Options.option(type = getattr(row, "OptionType"), 
#                         exp_date = getattr(row, "ExpirationDate"), 
#                         K = getattr(row, "Strike"),
#                         S0 = currentUnderlyingValue,
#                         r_benefit = r_benefit)

#             impVol = getattr(row, "impVol")
#             # updating the underlying value to get the new option value
#             opt.resetUnerlyingValue(underlyingValue=underlyingValue)
#             ttm = opt.getT(forwardDate) #base on passed in date
#             new_steps = americanBT.getDivT(forwardDate, getattr(row, "ExpirationDate"), payment_date)[1] * multi
#             new_divT = [americanBT.getDivT(forwardDate, getattr(row, "ExpirationDate"), payment_date)[0] * multi]

#             optValue = americanBT.americanBT(opt, ttm, rf, impVol, new_steps, div, new_divT)
#             PnL += (optValue - getattr(row, "CurrentPrice")) * getattr(row, "Holding")
#             print(PnL)
#     return PnL
def getPNL(portfolio, forwardDate, currentUnderlyingValue, underlyingValue, div):
    opt = Options.option(type = portfolio["OptionType"], 
                        exp_date =portfolio["ExpirationDate"], 
                        K = portfolio["Strike"],
                        S0 = currentUnderlyingValue,
                        r_benefit = r_benefit)
    impVol = portfolio["impVol"]
            # updating the underlying value to get the new option value
    opt.resetUnerlyingValue(underlyingValue=underlyingValue)
    ttm = opt.getT(forwardDate) #base on passed in date
    new_steps = americanBT.getDivT(forwardDate, portfolio["ExpirationDate"], payment_date)[1] * multi
    new_divT = [americanBT.getDivT(forwardDate, portfolio["ExpirationDate"], payment_date)[0] * multi]

    optValue = americanBT.americanBT(opt, ttm, rf, impVol, new_steps, div, new_divT)
    pnl = optValue - portfolio["CurrentPrice"]
    return pnl

portfolios = pd.read_csv("Week07\\problem2.csv")

#calculate implided volatilities
portfolios["impVol"] = 0
for row in portfolios.itertuples(name = "options"):
    if (getattr(row, "Type") == "Option"):
        opt = Options.option(type = getattr(row, "OptionType"), 
                        exp_date = getattr(row, "ExpirationDate"), 
                        K = getattr(row, "Strike"),
                        S0 = current_price,
                        r_benefit = r_benefit)
        steps = americanBT.getDivT(current_date, getattr(row, "ExpirationDate"), payment_date)[1] * multi
        divT = [americanBT.getDivT(current_date, getattr(row, "ExpirationDate"), payment_date)[0] * multi]

        impVol = americanBT.getImpVol(opt, current_date=current_date, rf = rf, value = getattr(row, "CurrentPrice"), steps = steps, div = div, divT = divT)
        portfolios.loc[row.Index, "impVol"] = impVol

# forawrd simulation
returns = pd.read_csv("Week07\\DailyReturn.csv")

r_AAPL = returns["AAPL"] 
std = r_AAPL.std() 

start = time.time()
# sumulate price in 10 days
nsim = 10000
p_arr = [] #to store simulated price
T = 10
simReturns = np.random.normal(loc = 0, scale = std, size = (T * nsim, 1))

for i in range(nsim):
    r = 1
    for j in range(T):
        r *= (1 + simReturns[T * i + j ])
    p_arr.append((current_price * r)[0])

forwardDate = "03/07/2022" # set the new date 10 days forward

portfolios_c = portfolios.copy()
portfolios_c.fillna("missing", inplace=True)
g2 = portfolios_c.groupby(["Type", "OptionType", "Strike", "CurrentPrice"])
pnl_summary = {}
df = []
i = 1
for name, optionGroup in g2:
    if(name[0] == "Option"):
        print(optionGroup)
        print(optionGroup.iloc[0,:].copy())
        pnl = []
        for price in p_arr:
            pnl.append(getPNL(optionGroup.iloc[0,:].copy(), forwardDate, current_price, price, div))
        # optionGroup["assetMEAN"] = np.mean(pnl) * optionGroup["Holding"]
        # optionGroup["assetVAR"] = getVaR.empirical(pnl) * optionGroup["Holding"]
        # optionGroup["assetES"] = getES.empirical(pd.DataFrame(pnl)) * optionGroup["Holding"]
        # print(optionGroup)
    if(name[0] == "Stock"):
        pnl = []
        for price in p_arr:
            pnl.append(price - current_price)
        # optionGroup["assetMEAN"] = np.mean(pnl)
        # optionGroup["assetVAR"] = getVaR.empirical(pnl)
        # optionGroup["assetES"] = getES.empirical(pd.DataFrame(pnl))
    dfPNL = pd.DataFrame(pnl)
    optionGroup["group"] = i
    df.append(optionGroup)
    pnl_summary[i] = dfPNL
    i+=1
portfolios_c = pd.concat(df)

groups = portfolios_c.groupby("Portfolio")


df_pnl = pd.DataFrame({"Mean": 0.0, "VaR(95%)": 0.0, "ES(95%)" : 0.0}, index =list(groups.groups.keys()))
for name, portfolio in groups:
    pnl_p = portfolio.iloc[0]["Holding"] * (pnl_summary[portfolio.iloc[0]["group"]].copy())
    n = portfolio.shape[0]

    for i in range(1, n):
        temp =  portfolio.iloc[i]["Holding"] * (pnl_summary[portfolio.iloc[i]["group"]].copy())
        pnl_p = pd.concat([pnl_p, temp], axis = 1)
    pnl_p["portfolioPNL"] = pnl_p.sum(axis = 1)
    
    df_pnl.loc[name, "Mean"] = pnl_p["portfolioPNL"].mean()
    df_pnl.loc[name, "VaR(95%)"] = getVaR.empirical(pnl_p["portfolioPNL"])
    df_pnl.loc[name, "ES(95%)"] = getES.empirical(pnl_p["portfolioPNL"])

print(df_pnl)
end = time.time()
print(end - start)

#calculate deltas
deltas = {}
impVols = {}
for name, portfolio in groups:
    deltas[name] = (getPortfolioDelta(portfolio, current_price, current_date, div))

df_deltas = pd.DataFrame(deltas, index=["Delta"]).T


df_deltas["VaR"] = st.norm.ppf(0.05, loc = 0, scale = std) * np.sqrt(10) * df_deltas["Delta"] * current_price 
df_deltas["ES"] = np.sqrt(10) * std * st.norm.pdf(st.norm.ppf(0.05)) / 0.05 * df_deltas["Delta"] * current_price 
print(df_deltas)
# df_pnl.join(df_deltas)
# print(df_pnl)
