import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from RiskMgmnt import Options
# def __init__(self, type,  exp_date, K, S0, r_benefit = 0, r_cost = 0, dateformat = '%m/%d/%Y'):


current_price = 164.85
rf = 0.0025
r_benefit = 0.0053
current_date = "02/25/2022"

portfolios = pd.read_csv("Week06\\problem3.csv")

groups = portfolios.groupby("Portfolio")

def getPortfolioValue(portfolio, underlyingValue, daysForward = 0):
    portfolioValue = 0
    PnL = 0
    #create a instance of the option class using data from each row
    for row in portfolio.itertuples(name = "options"):
        if (getattr(row, "Type") == "Stock"):
            portfolioValue += underlyingValue * getattr(row, "Holding")
            PnL += (underlyingValue - current_price) * getattr(row, "Holding")
        if (getattr(row, "Type") == "Option"):
            opt = Options.option(type = getattr(row, "OptionType"), 
                        exp_date = getattr(row, "ExpirationDate"), 
                        K = getattr(row, "Strike"),
                        S0 = current_price,
                        r_benefit = r_benefit)
            impVol = opt.getImpVol(current_date=current_date, rf = rf, value = getattr(row, "CurrentPrice"), daysForward = daysForward)
            # updating the underlying value to get the new option value
            opt.resetUnerlyingValue(underlyingValue=underlyingValue)
            optValue = opt.valueBS(current_date=current_date, sigma = impVol, rf = rf, daysForward = daysForward )[0]
            portfolioValue += optValue * getattr(row, "Holding")
            PnL += (optValue - getattr(row, "CurrentPrice")) * getattr(row, "Holding")
    return portfolioValue, PnL

dfs_value = []
dfs_PnL= []
for name, portfolio in groups:
    d_portfolioValue = {}
    d_PnL = {}
    for underlyingValue in range (140, 190):
        d_portfolioValue[underlyingValue] = getPortfolioValue(portfolio, underlyingValue=underlyingValue)[0]
        d_PnL[underlyingValue] = getPortfolioValue(portfolio, underlyingValue=underlyingValue)[1]

    df_portfolioValue = pd.DataFrame(d_portfolioValue, index = [name]).T
    df_PnL = pd.DataFrame(d_PnL, index = [name]).T

    dfs_value.append(df_portfolioValue)
    dfs_PnL.append(df_PnL)

summary_V = pd.concat(dfs_value, axis = 1)
summary_V.plot(figsize = (8, 8),
                 title = "Portfolio comparison(VALUE)",
                 xlabel = "Underlying Value",
                 ylabel = "Portfolio Value",
                 legend = 1)
plt.savefig("Week06\\plots\\Problem3_VALUE.png")

summary_PNL = pd.concat(dfs_PnL, axis = 1)
summary_PNL.plot(figsize = (8, 8),
                 title = "Portfolio comparison(PNL)",
                 xlabel = "Underlying Value",
                 ylabel = "Portfolio PNL",
                 legend = 1)
plt.axvline(x = current_price, ls = ':',color = 'r', label = "current market price")
plt.legend()
plt.savefig("Week06\\plots\\Problem3_PNL.png")


# forawrd simulation
returns = pd.read_csv("Week06\\DailyReturn.csv")
r_AAPL = returns["AAPL"]
# r_AAPL = returns["AAPL"] - returns["AAPL"].mean()
std = r_AAPL.std() 

nsim = 10000
p_arr = [] #to store simulated price
T = 10
simReturns = np.random.normal(loc = 0, scale = std, size = (T * nsim, 1))
# for i in range(nsim): #do nsim times of the simulation
#     r = np.random.normal(loc = 0, scale = std, size = (T, 1)) #draw 10 random returns from normal distribution
#     p = np.zeros([T, 1])
#     p[0] = current_price * (1 + r[0])
#     for i in range(1, T):
#         p[i] = p[i - 1] * (1 + r[i])

#     p10 = p[T - 1] #the price at day 10 in the future
#     p_arr.append(p10[0])

for i in range(nsim):
    r = 1
    for j in range(T):
        r *= (1 + simReturns[T * i + j ])
    p_arr.append(current_price * r)

dfs_value_sim = []
dfs_PnL_sim = []
dfs_PnL_sim_pct = []
for name, portfolio in groups:
    ls_portfolioValue_sim = []
    ls_PnL_sim = []
    ls_PnL_sim_pct = []
    current_value = getPortfolioValue(portfolio, underlyingValue=current_price, daysForward = 0)[0]
    for i in range (nsim):
        underlyingValue = float(p_arr[i])
        portfolioValue = getPortfolioValue(portfolio, underlyingValue=underlyingValue, daysForward = 10)[0] #daysForward = 10 
        ls_portfolioValue_sim.append(portfolioValue)
        portfolioPNL = getPortfolioValue(portfolio, underlyingValue=underlyingValue, daysForward = 10)[1]
        ls_PnL_sim.append(portfolioPNL)
        ls_PnL_sim_pct.append(portfolioPNL / current_value)
        
    df_portfolioValue_sim = pd.DataFrame({name: ls_portfolioValue_sim})
    df_PnL_sim = pd.DataFrame({name: ls_PnL_sim})
    df_PnL_sim_pct = pd.DataFrame({name: ls_PnL_sim_pct})

    dfs_value_sim.append(df_portfolioValue_sim)
    dfs_PnL_sim.append(df_PnL_sim)
    dfs_PnL_sim_pct.append(df_PnL_sim_pct)

summary_V_sim = pd.concat(dfs_value_sim, axis = 1)
summary_PNL_sim = pd.concat(dfs_PnL_sim, axis = 1)
summary_PNL_sim_pct = pd.concat(dfs_PnL_sim_pct, axis = 1)

from RiskMgmnt import getVaR, getES
MEAN = {}
VAR = {}
VAR_pct = {}
MEAN_pct = {}
ES = {}
ES_pct = {}
for name, portfolio_pnl in summary_PNL_sim.iteritems():
    MEAN[name] = portfolio_pnl.mean()
    VAR[name] = getVaR.empirical(portfolio_pnl)
    ES[name] = getES.empirical(portfolio_pnl)
for name, portfolio_pnl_pct in summary_PNL_sim_pct.iteritems():  
    MEAN_pct[name] = portfolio_pnl_pct.mean()  
    VAR_pct[name] = getVaR.empirical(portfolio_pnl_pct) * 100
    ES_pct[name] = getES.empirical(portfolio_pnl_pct) * 100

df_MEAN = pd.DataFrame(MEAN, index = ["Mean"]).T
df_VAR = pd.DataFrame(VAR, index = ["VaR"]).T
df_ES = pd.DataFrame(ES, index = ["ES"]).T
df_MEAN_pct = pd.DataFrame(MEAN_pct, index = ["Mean(%)"]).T
df_VAR_pct = pd.DataFrame(VAR_pct, index = ["VaR(%)"]).T
df_ES_pct = pd.DataFrame(ES_pct, index = ["ES(%)"]).T


data_sim = df_MEAN.join(df_VAR.join(df_ES))
print(data_sim)

data_sim.sort_values(by = "VaR", inplace = True)
data_sim_pct = df_VAR_pct.join(df_ES_pct.join(df_MEAN_pct))
data_sim_pct.sort_values(by = "VaR(%)", inplace = True)
# plt.cla()
# data_sim.plot(figsize = (10, 10), kind = "bar", title = "Portfolio Risk")
# plt.savefig("Week06\\plots\\Problem3_risk.png")

# print("Portfolio Risk\n", data_sim)

# print("Portfolio Risk (%)\n", data_sim_pct)
