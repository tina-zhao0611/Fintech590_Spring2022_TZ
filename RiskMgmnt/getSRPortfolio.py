import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def getAnnualReturn(DailyReturn):
    result = np.log((DailyReturn + 1) ** 255)
    return result


def getMinVol(r, sigma, r_target):
    def riskOptimize(w):
        w_m = np.matrix(w).T
        sigma_2 = (w_m.T * sigma * w_m)[0, 0]
        return sigma_2
    def getPortfolioReturn(w, r):
        w_m = np.matrix(w).T
        result = (w_m.T * r)[0,0]
        return result

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: getPortfolioReturn(w, r) - r_target},
            {'type': 'ineq', 'fun': lambda w: w - 0})

    result = minimize(riskOptimize, np.random.randn(20), constraints = cons)
    weight = result.x
    vol = riskOptimize(weight)
    return weight, vol


    

#df_r is the dataframe of assets returns
# columns: [stock_name, return]
# sigma is a np.matrix form of assets covariance
def getSRPortfolio(df_r, sigma, rf, r_down = 0.001, r_up = 0.2, freq = 50, nameColumnName = "stock_name", returnColumnName = "return"):
    stock_list = list(df_r[nameColumnName])
    r_stocks = np.matrix((df_r[returnColumnName]).values).T
    # plot a efficiency frontier
    rp = np.linspace(r_down, r_up, freq)
    vp = []
    for r in rp:
        weight, vol = getMinVol(r_stocks, sigma, r)
        vp.append(vol)
    
    optimal_sharpe = 0.0
    optimal_r = 0.0
    for i in range(len(rp)):
        c_sharpe = (rp[i] - rf) / np.sqrt(vp[i])
        if (c_sharpe > optimal_sharpe):
            optimal_sharpe = c_sharpe
            optimal_r = rp[i]
    optimal_weight, vol = getMinVol(optimal_r)
    marketPortfolio = pd.DataFrame({"Stock": stock_list,"weights(%)": [round(x, 4) for x in (optimal_weight * 100)]})

    return marketPortfolio


if __name__ == '__main__':
    import getReturn
    # stocks1 = pd.read_csv("Week07\\DailyReturn.csv")
    # stocks1.set_index("Date", drop = True, inplace=True)
    price = pd.read_csv("Week08\\updated_prices.csv")
    price.set_index("Date", drop = True, inplace=True)
    stocks2 = getReturn.return_calculate_mat(price)
    # stocks = pd.concat([stocks1, stocks2], axis = 0).dropna(axis = 1)
    
    sigma = np.matrix((np.log(stocks2 + 1)).cov()*255)

