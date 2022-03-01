import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import scipy.stats as st

import warnings
warnings.filterwarnings("ignore") 

from RiskMgmnt import getReturn, T_dist_fitter, Statistics, Simulations, getVaR, getES

prices = pd.read_csv("Week05\\DailyPrices.csv")
prices.drop(columns = "SPY", inplace = True) 
prices.set_index("Date", drop = True, append = True, inplace = True)

returns = getReturn.return_calculate_mat(prices, method = "ARITHMETIC")

# fit every stock's return into t distribution
stock_dist = pd.DataFrame(columns = ["ticker", "df", "scale"])
U = returns.copy()

for col_name, column in returns.iteritems():
    column = column - np.mean(column)
    df_t, mu_t, scale_t = T_dist_fitter.getT_MLE(column)
    stock_dist = stock_dist.append({'ticker': col_name, 'mu':mu_t, 'df':df_t, 'scale':scale_t}, ignore_index=True)
    
    # into uniform distribution through CDF
    temp_col = st.t.cdf(column, df = df_t, loc = mu_t, scale = scale_t)
    U.loc[:][col_name] = temp_col
stock_dist.set_index("ticker", drop = True, append = False, inplace = True)

# get the spearman correlation matrix
cor_mat = U.corr(method = "spearman")

# check the rank
Statistics.isPSD(cor_mat)

nsim = 5000
#do multivariant normal simulation
#and use cdf to convert into uniform distribution
simU = st.norm.cdf(Simulations.sim_pca(cor_mat, nsim, target = 1))
simU = pd.DataFrame(data = simU, columns = returns.columns)

simR = simU.copy()
#get simulated return by calculating quantaile using the fitted T for each column
for col_name, column in simU.iteritems():
    simR.loc[:][col_name] = st.t.ppf(column, df = stock_dist.loc[col_name]["df"], loc = stock_dist.loc[col_name]["mu"], scale = stock_dist.loc[col_name]["scale"])

#check the goodness of the simulation
print("distance between simulated returns and real returns (correlation(spearman) matrics): ", Statistics.fnorm(simR.corr(method = "spearman"), returns.corr(method = "spearman")))
print("distance between simulated returns and real returns (covariance matrics): ", Statistics.fnorm(np.cov(simR.T), np.cov(returns.T)))


holdings = pd.read_csv("Week05\\portfolio.csv")
iterations = pd.DataFrame({"iteration": np.array(range(nsim))})
groups = holdings.groupby("Portfolio")


# current prices of all stocks
current_price = prices.iloc[prices.shape[0] - 1, :]
current_price.index.name = "Stock" 
current_price = pd.DataFrame({"price": current_price}, index = current_price.index)

summary = pd.DataFrame() # for holding portfolio value simulations
for name, portfolio in groups:
    df_initial = pd.merge(portfolio, current_price, on = "Stock", how = "inner")
    initValue = sum(df_initial["Holding"] * df_initial["price"])

    simValue = []

    for i in range(nsim):
        iter_r = simR.iloc[i,:]
        iter_r.index.name = "Stock" 
        iter_r = pd.DataFrame({"return": iter_r}, index = iter_r.index)

        df_temp = pd.merge(df_initial, iter_r, on = "Stock", how = "inner")

        netValue = sum(df_temp["Holding"] * df_temp["price"] * (1 + df_temp["return"]))
        pnl = netValue - initValue
        simValue.append(pnl)

    summary[name] = simValue
summary["total"] = summary["A"] + summary["B"] + summary["C"]

# calculate VaR and ES 
for name, portfolio in summary.iteritems():
    print("Portfolio", name, "VaR: ", getVaR.empirical(portfolio, alpha = 0.05))
    print("Portfolio", name, "ES:  ", getES.empirical(portfolio, alpha = 0.05))
    