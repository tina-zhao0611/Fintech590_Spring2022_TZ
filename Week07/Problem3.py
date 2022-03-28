import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

    factor_return = {}
    T = factor_data.shape[0] 
    for f in factor_list:
        factor_return[f] = factor_data[f].mean()
    dailyRF = model_data["RF"].mean()
    dict_r = {}
    for r_name in stock_list:
        r_daily = 0
        for f in factor_list:
            r_daily += dict_param[r_name][f] * factor_return[f] + dailyRF
        r_annual = getAnnualReturn(r_daily)
        dict_r[r_name] = r_annual
    df_r = pd.DataFrame(dict_r, index = [0]).T
    df_r.columns = ["return"]
    return df_r


data = pd.read_csv("Week07\\F-F_Research_Data_Factors_daily.csv")
stocks = pd.read_csv("Week07\\DailyReturn.csv")
mom = pd.read_csv("Week07\\F-F_Momentum_Factor_daily.csv")
# eliminate white spaces
col_names = mom.columns.tolist()
for index,value in enumerate(col_names):
    col_names[index]= value.replace(" ","")
mom.columns=col_names 

stock_list = ["AAPL", "FB", "UNH", "MA",
                 "MSFT", "NVDA", "HD", "PFE",
                 "AMZN", "BRK-B", "PG", "XOM",
                 "TSLA", "JPM", "V", "DIS",
                 "GOOGL", "JNJ", "BAC", "CSCO"]

factor_list_FF3 = ["Mkt-RF", "SMB", "HML"]
factor_list_FFM = ["Mkt-RF", "SMB", "HML", "Mom"]


# convert the date format 
data["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), data["Date"]))
stocks["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(x, "%m/%d/%Y"), stocks["Date"]))
mom["Date"] = pd.Series(map(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d"), mom["Date"]))

# get past 10 years history data
startdate_str = "20120131"
data = data[data["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]
stocks = stocks[stocks["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]
mome = mom[mom["Date"]>datetime.datetime.strptime(startdate_str, "%Y%m%d")]

# divided by 100 to get like unit
data[factor_list_FF3] = data[factor_list_FF3]/100 
mom["Mom"] = mom["Mom"]/100

# align the dates, get the dataframe for the regressions
FF3 = pd.merge(stocks, data, on = "Date", how = "left")
FFM = pd.merge(FF3, mom, on = "Date", how = "left")

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

# solve for minmum variance for given return level
def getMinVol(r_target):
    def riskOptimize(w):
        w_m = np.matrix(w).T
        sigma_2 = (w_m.T * sigma * w_m)[0, 0]
        return sigma_2
    def getPortfolioReturn(w, r):
        w_m = np.matrix(w).T
        result = (w_m.T * r)[0,0]
        return result

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: getPortfolioReturn(w, r_stocks) - r_target},
            {'type': 'ineq', 'fun': lambda w: w - 0})

    result = minimize(riskOptimize, np.random.randn(20), constraints = cons)
    weight = result.x
    vol = riskOptimize(weight)
    return weight, vol


''' 
tried to use the cvxopt module but didn't work

# import cvxopt
# P = cvxopt.matrix(df_stocks.cov().values) 
# q = cvxopt.matrix(np.zeros(nStocks))
# A = cvxopt.matrix(np.append(np.ones(nStocks), np.array((exp_r_FF3["return"]/100).values)), (nStocks,2)).T
# b = cvxopt.matrix([1, r_target])
# G = cvxopt.matrix(-np.eye(nStocks))
# h = cvxopt.matrix(np.zeros(nStocks))
# result = cvxopt.solvers.coneqp(P,q,G,h,A,b)
# print(result['x']) 
'''

# plot a efficiency frontier
rp = np.linspace(0.001, 0.15, 50)
vp = []
for r in rp:
    weight, vol = getMinVol(r)
    w_m = np.matrix(weight).T
    result = (w_m.T * r_stocks)[0,0]
    # print(result)
    vp.append(vol)

plt.cla()
plt.plot(vp, rp)
plt.xlabel("Portfolio variance")
plt.ylabel("Portfolio annual return")
plt.title("Efficient Frontier")
plt.savefig("Week07\\plots\\Problem3_efficientFrontier")

#solve for the optimal portfolio
'''
wasn't able to get a stable convergence through this method

# rf = 0.0025
# def getSharpe(r):
#     weight, vol = getMinVol(r)
#     sharpe = (r - rf) / np.sqrt(vol)
#     return -sharpe

# cons = ({'type': 'ineq', 'fun': lambda r: r-rf})
# result2 = minimize(getSharpe, 0.01, constraints = cons)
'''
rf = 0.0025
optimal_sharpe = 0.0
optimal_r = 0.0
optimal_v = 0.0
d = []
for i in range(len(rp)):
    c_sharpe = (rp[i] - rf) / np.sqrt(vp[i])
    if (c_sharpe > optimal_sharpe):
        optimal_sharpe = c_sharpe
        optimal_r = rp[i]
        optimal_v = vp[i]
optimal_weight, vol = getMinVol(optimal_r)
marketPortfolio = pd.DataFrame({"Stock": stock_list,"weights(%)": [round(x, 4) for x in (optimal_weight * 100)]})
print(marketPortfolio)

plt.cla()
plt.plot(np.sqrt(vp), rp)
plt.plot([0, np.sqrt(optimal_v)], [rf, optimal_r])
plt.plot(np.sqrt(optimal_v), optimal_r,'or') 
plt.annotate('Market Portfolio', xy=(np.sqrt(optimal_v),optimal_r), xytext=(np.sqrt(optimal_v), optimal_r - 0.01),arrowprops=dict(arrowstyle='->'))
plt.xlabel("Portfolio variance")
plt.ylabel("Portfolio annual return")
plt.title("Efficient Frontier and Max SR portfolio")
plt.savefig("Week07\\plots\\Problem3_CML")
