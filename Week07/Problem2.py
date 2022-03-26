import sys,os

# __file__ = "Problem2.py"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from RiskMgmnt import getVaR, getES, Options, americanBT

# OPTION INFORMATION
current_price = 164.85
rf = 0.0025
r_benefit = 0.0053
current_date = "02/25/2022"

div = [1]
payment_date = "03/15/2022"

# forawrd simulation
returns = pd.read_csv("Week07\\DailyReturn.csv")

r_AAPL = returns["AAPL"] - returns["AAPL"].mean()
std = r_AAPL.std() 

# sumulate price in 10 days
nsim = 1000
p_arr = [] #to store simulated price
T = 10

for i in range(nsim): #do nsim times of the simulation
    r = np.random.normal(loc = 0, scale = std, size = (T, 1)) #draw 10 random returns from normal distribution
    p = np.zeros([T, 1])
    p[0] = current_price * (1 + r[0])
    for i in range(1, T):
        p[i] = p[i - 1] * (1 + r[i])

    p10 = p[T - 1] #the price at day 10 in the future
    p_arr.append(p10[0])


# mean VaR and ES of the stock price
stockMean = np.mean(p_arr)
stockVar = getVaR.empirical(p_arr, alpha = 0.05)
stockES = getES.empirical(pd.DataFrame(p_arr), alpha = 0.05)

# FUNCTION FOR CALCULATING PORTFOLIO PNL
def getPortfolioPnL(portfolio, currentDate, underlyingValue, steps, div, divT):
    PnL = 0
    for row in portfolio.itertuples(name = "options"):
        if (getattr(row, "Type") == "Stock"):
            PnL += (underlyingValue - current_price) * getattr(row, "Holding")
        if (getattr(row, "Type") == "Option"):
            opt = Options.option(type = getattr(row, "OptionType"), 
                        exp_date = getattr(row, "ExpirationDate"), 
                        K = getattr(row, "Strike"),
                        S0 = current_price,
                        r_benefit = r_benefit)
            impVol = americanBT.getImpVol(opt, current_date=current_date, rf = rf, value = getattr(row, "CurrentPrice"), steps = steps, div = div, divT = divT)
            # updating the underlying value to get the new option value
            opt.resetUnerlyingValue(underlyingValue=underlyingValue)
            ttm = opt.getT(currentDate) #base on passed in date
            optValue = americanBT.americanBT(opt, ttm, rf, impVol, steps, div, divT)
            PnL += (optValue - getattr(row, "CurrentPrice")) * getattr(row, "Holding")
    return PnL

portfolios = pd.read_csv("Week07\\problem2.csv")
groups = portfolios.groupby("Portfolio")

a = groups.get_group("ProtectedPut") 

currentDate = "03/07/2022" # set the new date 10 days forward
steps = 10
divT = [8] # set data based on dividend payment on 03/15

df_pnl = pd.DataFrame({"Mean": 0.0, "VaR": 0.0, "ES" : 0.0}, index =list(groups.groups.keys()))
for name, portfolio in groups:
    simulatePnl = []
    for price in p_arr:
        simulatePnl.append(getPortfolioPnL(portfolio, currentDate, price, steps, div, divT))
    Mean = np.mean(simulatePnl)
    Var = getVaR.normal(pd.DataFrame(simulatePnl), alpha = 0.05)
    ES = getES.normal(simulatePnl, alpha = 0.05)
    df_pnl.loc[name]["Mean"] = Mean
    df_pnl.loc[name]["VaR"] = Var
    df_pnl.loc[name]["ES"] = ES

print(df_pnl)
