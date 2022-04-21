import pandas as pd
import datetime
import numpy as np

import sys,os
from sympy import div
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

from RiskMgmnt import multiFactor, getOptimalPortfolio

def whiteSpace(df):
    col_names = df.columns.tolist()
    for index,value in enumerate(col_names):
        col_names[index]= value.replace(" ","")
    df.columns=col_names 
    return df        

data = pd.read_csv("Week07\\F-F_Research_Data_Factors_daily.csv")
stocks = pd.read_csv("Week07\\DailyReturn.csv")
mom = whiteSpace(pd.read_csv("Week07\\F-F_Momentum_Factor_daily.csv"))


factor_list = ["Mkt-RF", "SMB", "HML", "Mom"]
stock_list = ["AAPL", "FB", "UNH", "MA",
                 "MSFT", "NVDA", "HD", "PFE",
                 "AMZN", "BRK-B", "PG", "XOM",
                 "TSLA", "JPM", "V", "DIS",
                 "GOOGL", "JNJ", "BAC", "CSCO"]
data["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), data["Date"]))
stocks["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(x, "%m/%d/%Y"), stocks["Date"]))
mom["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), mom["Date"]))


data = pd.merge(data, mom, on = "Date", how = "left")
data[factor_list] =data[factor_list] / 100

toReg = pd.merge(stocks, data, how = "left")

betas, Betas = multiFactor.getParameters(toReg, stock_list, factor_list)

# get past 10 years history data
startdate_str = "20120131"
toMean = data[data["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]

stockMeans, covar, factorRegurn = multiFactor.getExpReturn(toMean, toReg, betas, stock_list, factor_list)

#portfolio construction
optimalWeithg, marketPortfolio, maxSharpe = getOptimalPortfolio.getWeights(stockMeans, covar, 0.0025, stock_list)


print(marketPortfolio)
print("sharpe = ", maxSharpe)

# # solve for minmum variance for given return level
# def getMinVol(r_target):
#     def riskOptimize(w):
#         w_m = np.matrix(w).T
#         sigma_2 = (w_m.T * sigma * w_m)[0, 0]
#         return sigma_2
#     def getPortfolioReturn(w, r):
#         w_m = np.matrix(w).T
#         result = (w_m.T * r)[0,0]
#         return result

#     cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
#             {'type': 'eq', 'fun': lambda w: getPortfolioReturn(w, r_stocks) - r_target},
#             {'type': 'ineq', 'fun': lambda w: w - 0})

#     result = minimize(riskOptimize, np.random.randn(20), constraints = cons)
#     weight = result.x
#     vol = riskOptimize(weight)
#     return weight, vol


# ''' 
# tried to use the cvxopt module but didn't work

# # import cvxopt
# # P = cvxopt.matrix(df_stocks.cov().values) 
# # q = cvxopt.matrix(np.zeros(nStocks))
# # A = cvxopt.matrix(np.append(np.ones(nStocks), np.array((exp_r_FF3["return"]/100).values)), (nStocks,2)).T
# # b = cvxopt.matrix([1, r_target])
# # G = cvxopt.matrix(-np.eye(nStocks))
# # h = cvxopt.matrix(np.zeros(nStocks))
# # result = cvxopt.solvers.coneqp(P,q,G,h,A,b)
# # print(result['x']) 
# '''

# # plot a efficiency frontier
# rp = np.linspace(0.005, 0.2, 50)
# vp = []
# for r in rp:
#     weight, vol = getMinVol(r)
#     w_m = np.matrix(weight).T
#     result = (w_m.T * r_stocks)[0,0]
#     # print(result)
#     vp.append(vol)

# plt.cla()
# plt.plot(vp, rp)
# plt.xlabel("Portfolio variance")
# plt.ylabel("Portfolio annual return")
# plt.title("Efficient Frontier")
# plt.savefig("Week07\\plots\\Problem3_efficientFrontier")

# #solve for the optimal portfolio
# '''
# wasn't able to get a stable convergence through this method

# # rf = 0.0025
# # def getSharpe(r):
# #     weight, vol = getMinVol(r)
# #     sharpe = (r - rf) / np.sqrt(vol)
# #     return -sharpe

# # cons = ({'type': 'ineq', 'fun': lambda r: r-rf})
# # result2 = minimize(getSharpe, 0.01, constraints = cons)
# '''
# rf = 0.0025
# optimal_sharpe = 0.0
# optimal_r = 0.0
# optimal_v = 0.0
# d = []
# for i in range(len(rp)):
#     c_sharpe = (rp[i] - rf) / np.sqrt(vp[i])
#     if (c_sharpe > optimal_sharpe):
#         optimal_sharpe = c_sharpe
#         optimal_r = rp[i]
#         optimal_v = vp[i]
# optimal_weight, vol = getMinVol(optimal_r)
# marketPortfolio = pd.DataFrame({"Stock": stock_list,"weights(%)": [round(x, 4) for x in (optimal_weight * 100)]})
# print(marketPortfolio)

# plt.cla()
# plt.plot(np.sqrt(vp), rp)
# plt.plot([0, np.sqrt(optimal_v)], [rf, optimal_r])
# plt.plot(np.sqrt(optimal_v), optimal_r,'or') 
# plt.annotate('Market Portfolio', xy=(np.sqrt(optimal_v),optimal_r), xytext=(np.sqrt(optimal_v), optimal_r - 0.01),arrowprops=dict(arrowstyle='->'))
# plt.xlabel("Portfolio variance")
# plt.ylabel("Portfolio annual return")
# plt.title("Efficient Frontier and Max SR portfolio")
# plt.savefig("Week07\\plots\\Problem3_CML")

