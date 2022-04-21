'''
module for calculating exponentially weighted covariance

function: cov_w(r, Lambda):

'''
import pandas as pd
import numpy as np

def expWeight(n, Lambda):
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = (1 - Lambda) * pow(Lambda, n -i + 1) 
    #the latest value is at the bottom
    #so generating weights accordingly (largest weight should be at the back)
    weights = weights / sum(weights)
    return weights

def cov_2arrays(x, y, w):
    cov = 0;
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    n = len(x)
    for i in range(n):
        cov += w[i] * (x[i] - mean_x) * (y[i] - mean_y)
    return cov

def cov_w(r, Lambda):
    n = r.size
    w = expWeight(n, Lambda)
    cov_w = np.sqrt(cov_2arrays(r.values, r.values, w))
    return cov_w

def expWeightedCov(df, nambda):
    n, m = df.shape
    weights = expWeight(n, nambda)
    Cov = pd.DataFrame(index = df.columns[1:], columns = df.columns[1:]) #the first column is the date
    for i in range(m - 1):
        for j in range(m - 1):
            Cov.iloc[i][j] = cov_2arrays(df.iloc[:, i + 1], df.iloc[:, j + 1], weights)
    return Cov

if __name__ == '__main__':
    weight = expWeight(20, 0.95)
    print(weight)