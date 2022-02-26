'''
module for calculating VES

2 methods are implemented

fitting normal distribution
fitting generalized t distribution with MLE

return the value of ES (negative value means a loss)

'''

import numpy as np
import pandas as pd
import scipy.stats as st


#Fitting normal distribution
def normal(data, alpha = 0.05):
    data = pd.DataFrame({"x": data})
    mu = data["x"].mean()
    data -= mu
    sigma = np.std(data["x"])

   
    ES_Normal = -mu + sigma * (st.norm.pdf(st.norm.ppf(alpha)))/alpha
 
    return -ES_Normal

#Fitting T distribution
def T(data, alpha = 0.05):
    data = pd.DataFrame({"x": data})
    mu = data["x"].mean()
    t_df, t_m, t_s = st.t.fit(data["x"])
    
    # simulate t distribution with estimated parameters
    t = st.t.rvs(df = t_df, loc = t_m, scale = t_s, size = 10000)
    tsim = pd.DataFrame({"tsim": t})

    VaR_T = np.percentile(tsim, alpha)
    temp = tsim[tsim <= VaR_T].dropna()
    ES_T = temp["tsim"].mean() + mu
    
    return ES_T
