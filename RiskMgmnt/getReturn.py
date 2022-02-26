'''
module for calculating return

4 methods can be chose from

"BM": Classical Brownian Motion
"ARITHMETIC"(default): Arithmetic Return
"GBM": Geometric Brownian Motion

'''
import pandas as pd
import numpy as np

def return_calculate(price, method = "ARITHMETIC", column_name = "price"):
    T = price.shape[0]
    price["p1p0"] = 0

    for i in range(1, T):
        #Classical Brownian Motion
        if(method == "BM"):
            price.loc[i, "p1p0"] = price.loc[i, column_name]  - price.loc[i - 1, column_name] 

        #If other two methods
        else:
            price.loc[i, "p1p0"] = price.loc[i, column_name]  / price.loc[i - 1, column_name] 

    df_r = price[1:].copy()

    #Classical Brownian Motion
    if(method == "BM"):
        df_r.loc[:, "return"] = df_r.loc[:, "p1p0"]

    #Arithmetic Return
    if(method == "ARITHMETIC"):
        df_r.loc[:, "return"] = df_r.loc[:, "p1p0"] - 1

    #Geometric Brownian Motion
    if(method == "GBM"):
        df_r.loc[:, "return"] = np.log(df_r.loc[:, "p1p0"])

    df_r = df_r.drop(columns="p1p0")
    
    return df_r