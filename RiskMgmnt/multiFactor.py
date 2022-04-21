import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np


def getParameters(toReg, asset_list, factor_list):

    X = np.array(sm.add_constant(toReg[factor_list]))
    Y =np.array(toReg[asset_list])

    Betas = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)[1:]
    result = pd.DataFrame(Betas, index = factor_list, columns = asset_list)
    return result, Betas

def getExpReturn(factorToMean, toReg, betas, asset_list, factor_list):
    
    Y =np.array(toReg[asset_list])

    factorReturn = factorToMean.mean(axis = 0)[factor_list]
    stockMeans = np.matrix(np.log(1 + np.dot(betas.T, factorReturn)) * 255)
    covar = np.matrix((np.cov(np.log(Y.T + 1)))*255)
    return stockMeans, covar, factorReturn


