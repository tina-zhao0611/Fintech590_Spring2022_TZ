import pandas as pd
import numpy as np
from scipy.optimize import minimize


def getWeights(stockMeans, covar, rf, nameList):

    # stockMeans = np.matrix(returnMax)
    # covar = np.matrix(covMax)

    nStocks = stockMeans.shape[1]


    def sharpeCalculate(w, stockReturn, stockCov, rf):
        w = np.matrix(w)
        r_p = w * stockReturn.T
        s_p = np.sqrt(w * stockCov * w.T)
        sharpe = (r_p[0,0] - rf) / s_p[0,0] 
        return -sharpe
    x0 = np.array(nStocks*[1 / nStocks])
    args = (stockMeans, covar, rf)
    bound = [(0.0, 1) for _ in nameList]
    cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
    result = minimize(sharpeCalculate, x0 = x0, args = args, bounds = bound, constraints = cons)

    optimal_weight = result.x
    marketPortfolio = pd.DataFrame({"Stock": nameList,"weights(%)": [round(x, 4) for x in (optimal_weight * 100)]})
    print(marketPortfolio)
    
    return optimal_weight