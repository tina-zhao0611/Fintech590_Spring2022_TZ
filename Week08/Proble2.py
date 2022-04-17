import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from RiskMgmnt import getReturn

def whiteSpace(df):
    col_names = df.columns.tolist()
    for index,value in enumerate(col_names):
        col_names[index]= value.replace(" ","")
    df.columns=col_names 
    return df

data = whiteSpace(pd.read_csv("Week08\\F-F_Research_Data_Factors_daily.csv"))
stocks = whiteSpace(pd.read_csv("Week08\\DailyReturn.csv"))
mom = whiteSpace(pd.read_csv("Week08\\F-F_Momentum_Factor_daily.csv"))

factor_list = ["Mkt-RF", "SMB", "HML", "Mom"]
stock_list = ["AAPL",  "MSFT", "BRK-B", "JNJ", "CSCO"]

data["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), data["Date"]))
stocks["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(x, "%m/%d/%Y"), stocks["Date"]))
mom["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), mom["Date"]))

FFM = pd.merge(data, mom, on = "Date", how = "left").dropna(axis = 0)

# divided by 100 to get like unit
FFM[factor_list] = FFM[factor_list]/100 
toReg = pd.merge(stocks, FFM, on = "Date", how = "left").dropna(axis = 0)

X = np.array(sm.add_constant(toReg[factor_list]))
Y =np.array(toReg[stock_list])

Betas = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)[1:]

startdate_str = "20120114"
ENDdate_str = "20220115"
toMean = FFM[FFM["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]
toMean = toMean[toMean["Date"]<datetime.datetime.strptime(ENDdate_str, "%Y%m%d")]

factorReturn = toMean.mean(axis = 0)[factor_list]
stockMeans = np.matrix(np.log(1 + np.dot(Betas.T, factorReturn)) * 255)
covar = np.matrix((np.cov(np.log(Y.T + 1)))*255)

nStocks = len(stock_list)


def sharpeCalculate(w, stockReturn, stockCov, rf):
    w = np.matrix(w)
    r_p = w * stockReturn.T
    s_p = np.sqrt(w * stockCov * w.T)
    sharpe = (r_p[0,0] - rf) / s_p[0,0] 
    return -sharpe
x0 = np.array(nStocks*[1 / nStocks])
args = (stockMeans, covar, 0.00025)
bound = [(0.0, 1) for _ in stock_list]
cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
result = minimize(sharpeCalculate, x0 = x0, args = args, bounds = bound, constraints = cons)

optimal_weight = result.x
marketPortfolio = pd.DataFrame({"Stock": stock_list,"weights(%)": [round(x, 4) for x in (optimal_weight * 100)]})
print(marketPortfolio)


updateP = pd.read_csv("Week08\\updated_prices.csv")
updateR = getReturn.return_calculate_mat(updateP)
updateR.reset_index(drop = True, inplace = True)

updateFF3 = whiteSpace(pd.read_csv("Week08\\updated_F-F_Research_Data_Factors_daily.csv"))
updatemom = whiteSpace(pd.read_csv("Week08\\updated_F-F_Momentum_Factor_daily.csv"))


updateFF3["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), updateFF3["Date"]))
updatemom["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), updatemom["Date"]))
updateR["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%m/%d/%Y"), updateR["Date"]))

updateFFM = pd.merge(updateFF3, updatemom, on = "Date", how = "left").dropna(axis = 0)
updateFFM[factor_list] = updateFFM[factor_list] / 100

allFFM = pd.concat([FFM, updateFFM], axis = 0)
upData = pd.merge(updateR, allFFM, on = "Date", how = "left").dropna(axis = 0)

initialWeight = np.array([0.1007598818153811, 0.2095098186253345, 0.43839111238558587, 0.17015442982085535, 0.08118475735284322])
lastW= initialWeight
# lastW = optimal_weight
R = updateR.copy().reset_index()[stock_list]
ffReturns = upData[factor_list] 
t = R.shape[0]

weightList = []
factorW = np.matrix(sum((Betas*initialWeight).T))

pReturn = []
residR = []
for i in range(t):
    weightList.append(lastW)
    lastW = np.array(lastW * (1 + R.iloc[i,:])) 
    sumW = sum(lastW)
    lastW = lastW / sumW
    pReturn.append(sumW - 1)
    rR = (sumW - 1) - factorW * np.matrix(ffReturns.iloc[i,:]).T
    residR.append(rR[0,0])
    

pReturn = np.array(pReturn)
weights = pd.DataFrame(weightList, columns = stock_list) 
totalReturn = np.exp(sum(np.log(pReturn + 1))) - 1
upData.insert(loc = upData.shape[1], column = "Alpha", value = np.array(residR))


K = np.log(totalReturn + 1) / totalReturn
carinoK = (np.log(pReturn + 1) / pReturn)/K

factorWeights = factorW.repeat(t, 0).reshape(t, len(factor_list))
r_carinoK = carinoK.repeat(4).reshape(factorWeights.shape)
Attrib = ffReturns * factorWeights * r_carinoK

residCol = residR * carinoK
Attrib.insert(loc = Attrib.shape[1], column = "Alpha", value = residCol)

new_factor_list = factor_list + ["Alpha"]

TR = []
for col in upData[new_factor_list].iteritems():
    tr = np.exp(sum(np.log(col[1] + 1))) - 1
    TR.append(tr)
print(TR)

ATR = []
for col in Attrib.iteritems():
    ATR.append(sum(col[1]))
print(ATR)

Attribution = pd.DataFrame({"TotalReturn": TR, "ReturnAttribution": ATR}, index = new_factor_list)


Y = ffReturns * factorWeights
Y.insert(loc = Y.shape[1], column = "Alpha", value = np.array(residR))
X =  np.array(sm.add_constant(pd.DataFrame({"pReturn": pReturn})))
Y = np.array(Y)
Beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)[1]
cSD = Beta * np.std(pReturn, ddof = 1)
Attribution.insert(loc = 2, column = "VolAttribution", value = cSD)
