import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
from numpy.linalg import eigh
from numpy.linalg import norm


# testing the covariance estimate module
def test_covarianc():
    from RiskMgmnt import covEstimate

    df = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})
    print(covEstimate.cov_manual(df["x"].copy(), df["y"].copy()))
    print(covEstimate.pearson_manual(df["x"].copy(), df["y"].copy()))
    print(covEstimate.spearman_manual(df["x"].copy(), df["y"].copy()))

# testing the covariance estimate module
def test_psd():
    from RiskMgmnt import getPSD

    n=500
    sigma = np.zeros([n, n]) + 0.9
    for i in range(n):
        sigma[i, i] = 1.0
    sigma[0,1] = 0.7357
    sigma[1,0] = 0.7357

    if (sum(eigh(sigma)[0]>-1e-8) < n): #original matrix is not psd    
        print("tunning near_psd")
        sigma_psd = getPSD.near_psd(sigma)
        print("tunning Higham_psd")
        higham_psd = getPSD.higham_psd(sigma)

    if(sum(eigh(sigma_psd)[0]>-1e-8) == n):
        print("near_psd succeed")
    if(sum(eigh(higham_psd)[0]>-1e-8) == n):
        print("higham_psd succeed")
    

# testing the simulation module
def test_simulation():
    from RiskMgmnt import Simulations

    data = pd.read_csv("Week03\\DailyReturn.csv")
    cov_mat = data.cov().values
    nsim = 25000

    s_direct = Simulations.sim_direct(cov_mat, nsim)
    s_pca = Simulations.sim_pca(cov_mat, nsim, 0.75)

    cov_direct = np.cov(s_direct.T)
    diff_direct = norm(cov_direct - cov_mat, 'fro')
    cov_pca = np.cov(s_pca.T)
    diff_pca = norm(cov_pca - cov_mat, 'fro')

    print(diff_direct)
    print(diff_pca)

# testing the VaR module
def test_var():
    from RiskMgmnt import getVaR
    
    #use problem1's data for testing
    data = pd.read_csv("Week05\\problem1.csv")
    
    VaR_Normal = getVaR.normal(data.copy()) 
    print("VaR_Normal: ", VaR_Normal)
    
    VaR_Normal_w = getVaR.normal_w(data.copy()) 
    print("VaR_Normal_weighted: ", VaR_Normal_w)
    
    VaR_t_mle = getVaR.t_mle(data.copy()) 
    print("VaR_t_mle: ", VaR_t_mle)
    
    VaR_hist = getVaR.hist(data.copy()) 
    print("VaR_hist: ", VaR_hist)
    
    VaR_T = getVaR.T(data.copy()) 
    print("VaR_T: ", VaR_T)
    
#testing the ES module
def test_es():
    from RiskMgmnt import getES
     
    data = pd.read_csv("Week05\\problem1.csv")
    ES_Normal = getES.normal(data["x"].copy()) 
    print("ES_Normal: ", ES_Normal)
    
    ES_T = getES.T(data["x"].copy())
    print("ES_T: ", ES_T)

# test_covarianc()
# test_psd()
# test_simulation()
test_var()
test_es()