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
from RiskMgmnt import getReturn

def getAnnualReturn(DailyReturn):
    result = np.log((DailyReturn + 1) ** 255)
    return result
        
def getEstimateReturn(stock_list, model_data, factor_data, factor_list):
    dict_param = {}

    for r_name in stock_list:
        regression = sm.OLS((model_data[r_name] - model_data["RF"]), sm.add_constant(model_data[factor_list])) 
        ols_model = regression.fit()
        # OLS:
        #   y: r - rf constant: 
        #   alpha 
        #   x1~xk: factor returns
        
        # print(ols_model.summary())
        param = {}
        # param["alpha"] = ols_model.params["const"] don't need to record alpha
        for f in factor_list:
            param[f] = ols_model.params[f] 
        dict_param[r_name] = param
        print(dict_param)

    factor_return = {}
    T = factor_data.shape[0] 
    for f in factor_list:
        factor_return[f] = factor_data[f].mean()
    dailyRF = model_data["RF"].mean()
    dict_r = {}
    for r_name in stock_list:
        r_daily = 0
        for f in factor_list:
            r_daily += dict_param[r_name][f] * factor_return[f]

            # r_daily += dict_param[r_name][f] * factor_return[f] + dailyRF
        r_annual = getAnnualReturn(r_daily)
        dict_r[r_name] = r_annual
    df_r = pd.DataFrame(dict_r, index = [0]).T
    df_r.columns = ["return"]
    return df_r


data = pd.read_csv("Week08\\F-F_Research_Data_Factors_daily.csv")
stocks = pd.read_csv("Week08\\DailyReturn.csv")
mom = pd.read_csv("Week08\\F-F_Momentum_Factor_daily.csv")
# eliminate white spaces
col_names = mom.columns.tolist()
for index,value in enumerate(col_names):
    col_names[index]= value.replace(" ","")
mom.columns=col_names 

stock_list = ["AAPL",  "MSFT", "BRK-B", "JNJ", "CSCO"]

factor_list_FF3 = ["Mkt-RF", "SMB", "HML"]
factor_list_FFM = ["Mkt-RF", "SMB", "HML", "Mom"]


# convert the date format 
data["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), data["Date"]))
stocks["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(x, "%m/%d/%Y"), stocks["Date"]))
mom["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), mom["Date"]))
# divided by 100 to get like unit
data[factor_list_FF3] = data[factor_list_FF3]/100 
mom["Mom"] = mom["Mom"]/100
# align the dates, get the dataframe for the regressions
FF3 = pd.merge(stocks, data, on = "Date", how = "left")
FFM = pd.merge(FF3, mom, on = "Date", how = "left")

startdate_str = "20120114"
ENDdate_str = "20220115"
data = data[data["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]
stocks = stocks[stocks["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]
mom = mom[mom["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]
data = data[data["Date"]<datetime.datetime.strptime(ENDdate_str, "%Y%m%d")]
stocks = stocks[stocks["Date"]<datetime.datetime.strptime(ENDdate_str, "%Y%m%d")]
mom = mom[mom["Date"]<datetime.datetime.strptime(ENDdate_str, "%Y%m%d")]




factor_data_FF3 = data.copy()
factor_data_FFM = pd.merge(factor_data_FF3, mom).copy()

exp_r_FF3 = getEstimateReturn(stock_list, model_data = FF3, factor_data = factor_data_FF3, factor_list = factor_list_FF3)
exp_r_FFM = getEstimateReturn(stock_list, model_data = FFM, factor_data = factor_data_FFM, factor_list = factor_list_FFM)

summary = exp_r_FF3.rename(columns={'return': 'estimated annual return by FF3 model (%)'}).join(exp_r_FFM.rename(columns={'return': 'estimated annual return by FF-M model (%)'}))
print(summary*100)

#portfolio construction
nStocks = len(stock_list)
df_stocks = stocks[stock_list]

sigma = np.matrix((np.log(df_stocks + 1)).cov()*255)
r_stocks = np.matrix((exp_r_FFM["return"]).values).T

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
#     def getSharpe(w):
#         sharpe = 

#     cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
#             {'type': 'eq', 'fun': lambda w: getPortfolioReturn(w, r_stocks) - r_target},
#             {'type': 'ineq', 'fun': lambda w: w - 0})

#     result = minimize(riskOptimize, np.random.randn(nStocks), constraints = cons)
#     weight = result.x
#     vol = riskOptimize(weight)
#     return weight, vol



# # plot a efficiency frontier
# rp = np.linspace(0.06, 0.15, 60)
# vp = []
# for r in rp:
#     weight, vol = getMinVol(r)
#     w_m = np.matrix(weight).T
#     result = (w_m.T * r_stocks)[0,0]
#     # print(result)
#     vp.append(vol)

# # plt.cla()
# # plt.plot(vp, rp)
# # plt.xlabel("Portfolio variance")
# # plt.ylabel("Portfolio annual return")
# # plt.title("Efficient Frontier")
# # plt.savefig("Week08\\plots\\Problem3_efficientFrontier")

# #solve for the optimal portfolio
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


stock_list = ["AAPL",  "MSFT", "BRK-B", "JNJ", "CSCO"]
initialWeight = np.array([0.1007598818153811, 0.2095098186253345, 0.43839111238558587, 0.17015442982085535, 0.08118475735284322])
# intialWeight = pd.DataFrame({"stock": stock_list, "weight": Weight})

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
print(Attribution)