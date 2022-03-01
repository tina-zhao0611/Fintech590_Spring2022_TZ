import numpy as np
import scipy.stats as st
from scipy.optimize import minimize

def likelyhood_t(parameter, x):
    n = parameter[0]
    mu = parameter[1]
    std = parameter[2]
    
    L = np.sum(st.t.logpdf(x, df = n, loc = mu, scale = std)) 
    return -L

def getT_MLE(r):
    cons = ({'type': 'ineq', 'fun': lambda x: x[2] - 0}) #standard deviation should be larger than 0
    mle_model = minimize(likelyhood_t, [2, r.mean(), r.std()], args = r, constraints = cons) #minimizing -L means maximizing L
    df_t = mle_model.x[0]
    mu_t = mle_model.x[1]
    scale_t = mle_model.x[2]
    return df_t, mu_t, scale_t

