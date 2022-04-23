from cvxopt import mul
import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import sys,os

from sympy import div
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

from RiskMgmnt import getReturn, multiFactor, getOptimalPortfolio


data = pd.read_csv("Week08\\F-F_Research_Data_Factors_daily.csv")
stocks = pd.read_csv("Week08\\DailyReturn.csv")
mom = pd.read_csv("Week08\\F-F_Momentum_Factor_daily.csv")
# eliminate white spaces
col_names = mom.columns.tolist()
for index,value in enumerate(col_names):
    col_names[index]= value.replace(" ","")
mom.columns=col_names 

stock_list = ["AAPL",  "MSFT", "BRK-B", "JNJ", "CSCO"]

factor_list = ["Mkt-RF", "SMB", "HML", "Mom"]


# convert the date format 
data["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), data["Date"]))
stocks["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(x, "%m/%d/%Y"), stocks["Date"]))
mom["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), mom["Date"]))

toReg = pd.merge(stocks, pd.merge(data, mom, on = "Date", how = "left"), on = "Date", how = "left")
toReg[factor_list] = toReg[factor_list] /100

betas, Betas = multiFactor.getParameters(toReg, stock_list, factor_list)

print(betas)
print(Betas)

startdate_str = "20120114"
ENDdate_str = "20220115"
data = data[data["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]
mom = mom[mom["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]
data = data[data["Date"]<datetime.datetime.strptime(ENDdate_str, "%Y%m%d")]
mom = mom[mom["Date"]<datetime.datetime.strptime(ENDdate_str, "%Y%m%d")]

toMean = pd.merge(data, mom, on = "Date", how = "left")
print(toMean)

stockMeans, covar, factorReturn = multiFactor.getExpReturn(toMean, toReg, Betas, stock_list, factor_list)
optWeight, marketPortfolio, maxSharpe = getOptimalPortfolio.getWeights(stockMeans, covar, 0.0025, stock_list)

print(factorReturn)
print(marketPortfolio)

#portfolio construction
initialWeight = np.array([0.1007598818153811, 0.2095098186253345, 0.43839111238558587, 0.17015442982085535, 0.08118475735284322])
# intialWeight = pd.DataFrame({"stock": stock_list, "weight": Weight})
print("initial:",initialWeight)
print("opt", optWeight)
updateP = pd.read_csv("Week08\\updated_prices.csv")
updateR = getReturn.return_calculate_mat(updateP)
R = updateR.copy().reset_index()[stock_list]

weightList = []

lastW = initialWeight
pReturn = []
t = R.shape[0]
for i in range(t):
    weightList.append(lastW)
    lastW = np.array(lastW * (1 + R.iloc[i,:])) 
    sumW = sum(lastW)
    lastW = lastW / sumW
    pReturn.append(sumW - 1)

pReturn = np.array(pReturn)
pstd= np.std(pReturn, ddof = 1)
weights = pd.DataFrame(weightList, columns = stock_list) 
totalReturn = np.exp(sum(np.log(pReturn + 1))) - 1

K = np.log(totalReturn + 1) / totalReturn
carinoK = (np.log(pReturn + 1) / pReturn)/K

TR = []
for col in R.iteritems():
    tr = np.exp(sum(np.log(col[1] + 1))) - 1
    TR.append(tr)
print(TR)

ATR = []
Y = R * weights 
for col in Y.iteritems():
    newCol = col[1] * carinoK
    ATR.append(sum(newCol))
print(ATR)
    
Attribution = pd.DataFrame({"TotalReturn": TR, "ReturnAttribution": ATR}, index = stock_list)

X =  np.array(sm.add_constant(pd.DataFrame({"pReturn": pReturn})))
Y = np.array(Y)
Beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)[1]
cSD = Beta * np.std(pReturn, ddof = 1)
Attribution.insert(loc = 2, column = "VolAttribution", value = cSD)
portfolio = pd.DataFrame({"TotalReturn": totalReturn, "ReturnAttribution": totalReturn, "VolAttribution": pstd}, index = ["Portfolio"])
Attribution = pd.concat([Attribution, portfolio], axis = 0)
print(Attribution) 