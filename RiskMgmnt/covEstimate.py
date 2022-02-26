"""
module for the covariance & coefficient estimations
all functions can operate on both DataFrame columns and arrays

func1 : cov_manual(x, y)
    calculate covariance between x and y manually.
    results varified to be the same as provided by "np.cov(x,y)"

func2 : pearson_manual(x, y)
    calculate the pearson coefficient between x and y manually.
    results varified to be the same as provided by "scipy.stats.pearsonr(x, y)"
    
func2 : spearman_manual(x, y)
    calculate the spearman coefficient between x and y manually.
    results varified to be the same as provided by "scipy.stats.spearmanr(x, y)"
"""

from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd

def cov_manual(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    n = df.shape[0]
    df["temp"] = (df["x"] - df["x"].mean()) * (df["y"] - df["y"].mean())
    cov = df['temp'].sum() / (n-1)
    # print("calculated:", cov)
    # print("real:", np.cov(x,y)[0,1])
    return cov
    
def pearson_manual(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    cov = cov_manual(x, y)
    rho_pearson = cov / (df["x"].std() * df["y"].std())
    # print("calculated:", rho_pearson)
    # print("real:", pearsonr(df['x'], df['y'])[0])
    return rho_pearson


def spearman_manual(x, y): #Use Pearson coefficient on the ranks
    df = pd.DataFrame({'x': x, 'y': y})
    df['x_rank'] = df['x'].rank()
    df['y_rank'] = df['y'].rank()
    rho_spearman, p = pearsonr(df['x_rank'], df['y_rank'])
    # print("calculated:", rho_spearman)
    # print("real:", spearmanr(df['x'], df['y'])[0])
    return rho_spearman

#data used to verify function results
# df = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})
# cov_manual(df["x"].copy(), df["y"].copy())
# pearson_manual(df["x"].copy(), df["y"].copy())
# spearman_manual(df["x"].copy(), df["y"].copy())

