import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sympy import N
import scipy.stats as st
from tqdm import tqdm


import sys,os
from sympy import div
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

from RiskMgmnt import getReturn, multiFactor, getOptimalPortfolio, T_dist_fitter,Statistics, Simulations, getES, getVaR

def whiteSpace(df):
    col_names = df.columns.tolist()
    for index,value in enumerate(col_names):
        col_names[index]= value.replace(" ","")
    df.columns=col_names 
    return df

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


ffm = pd.merge(data, mom, on = "Date", how = "left")

toReg = pd.merge(stocks, pd.merge(data, mom, on = "Date", how = "left"), on = "Date", how = "left")
toReg[factor_list] = toReg[factor_list] /100

betas, Betas = multiFactor.getParameters(toReg, stock_list, factor_list)

print(betas)

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


optW = np.array([0.1007598818153811, 0.2095098186253345, 0.43839111238558587, 0.17015442982085535, 0.08118475735284322])
# optW = optWeight

def pVol(w):
    pvol = (w.T * covar * w)[0, 0]
    return np.sqrt(pvol)

def pCSD(w):
    pvol = pVol(w)
    csd = np.multiply(np.multiply(w, np.dot(covar, w)), 1/pvol)
    return csd    

def riskBudget(w):
    pSig = pVol(w)
    CSD = pCSD(w)
    rb = np.multiply(CSD, 1/pSig)
    return rb

riskBudgetOpt = pd.DataFrame(riskBudget(np.matrix(optW).T), index = stock_list)
    
def sseCSD(w):
    w = np.matrix(w).T
    n = w.shape[0]
    csd = pCSD(w)
    mCSD = np.sum(csd) / n
    dCSD = csd - mCSD
    se = np.multiply(dCSD,dCSD)
    return np.sum(se) * 100000

nStocks = len(stock_list)
x0 = np.array(nStocks*[1 / nStocks])
bound = [(0.0, 1) for _ in stock_list]
cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
result = minimize(sseCSD, x0 = x0, bounds = bound, constraints = cons)

riskParityW = result.x

riskBudget(np.matrix(riskParityW).T)

# remove mean
Y =np.array(toReg[stock_list])

m = np.mean(Y, 0)
Y_m = Y - m

models = pd.DataFrame(columns=["df","mu","scale"], index = stock_list)
# for i in range(Y_m.shape[1]):
#     r = Y_m[:, i]
#     df_t, mu_t, scale_t = T_dist_fitter.getT_MLE(r)
#     models.iloc[i,:] = [df_t, mu_t, scale_t]
Y_m_df = pd.DataFrame(data = Y_m, columns = stock_list)
U = Y_m_df.copy()

i = 0
for col_name, column in Y_m_df.iteritems():
    column = column - np.mean(column)
    df_t, mu_t, scale_t = T_dist_fitter.getT_MLE(column)
    models.iloc[i,:] = [df_t, mu_t, scale_t]
    # into uniform distribution through CDF
    temp_col = st.t.cdf(column, df = df_t, loc = mu_t, scale = scale_t)
    U.loc[:][col_name] = temp_col
    i += 1
    
nsim = 50000

cor_mat = U.corr(method = "spearman")

# check the rank
Statistics.isPSD(cor_mat)

#do multivariant normal simulation
#and use cdf to convert into uniform distribution
simU = st.norm.cdf(Simulations.sim_pca(cor_mat, nsim, target = 1))
simU = pd.DataFrame(data = simU, columns = Y_m_df.columns)

simR = simU.copy()
#get simulated return by calculating quantaile using the fitted T for each column
for col_name, column in simU.iteritems():
    simR.loc[:][col_name] = st.t.ppf(column, df = models.loc[col_name]["df"], loc = models.loc[col_name]["mu"], scale = models.loc[col_name]["scale"])

def pES(w):
    r = np.dot(simR, w.T)
    r = pd.DataFrame(r)
    es = getES.empirical(r)
    return es

def CES(w):
    x = w
    n = w.shape[0]
    es = pES(w)
    ces = np.zeros(n)
    e = 1e-6
    for i in range(n):
        old = x[i]
        x[i] = x[i] + e
        ces[i] = old * (pES(x) - es) / e
        x[i] = old
    return ces

def sseCES(w):
    ces = CES(w)
    ces_m = ces - np.mean(ces)
    return np.dot(ces_m.T, ces_m) * 10000

nStocks = len(stock_list)
x0 = np.array(nStocks*[1 / nStocks])
bound = [(0.0, 1) for _ in stock_list]
cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
result = minimize(sseCES, x0 = x0, bounds = bound, constraints = cons)

ESriskParityW = result.x

summ = pd.DataFrame({"volParity":riskParityW, "ESParity": ESriskParityW}, index = stock_list)
print(summ)
