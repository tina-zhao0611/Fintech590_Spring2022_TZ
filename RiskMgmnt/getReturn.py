'''
module for calculating return
    return_calculate() operate on a specified column
    return_calculate_mat() will calculate returns on every column of the DataFrame provided

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
            price.loc[i, "p1p0"] = price.loc[i, column_name].values  / price.loc[i - 1, column_name].values 

    df_r = price[1:].copy()

    #Classical Brownian Motion
    if(method == "BM"):
        df_r.loc[:, "return"] = df_r.loc[:, "p1p0"]

    #Arithmetic Return
    if(method == "ARITHMETIC"):
        df_r.loc[:, "return"] = df_r.loc[:, "p1p0"] - 1

    #Geometric Brownian Motion
    if(method == "GBM"):
        df_r.loc[:, "return"] = np.log(df_r.loc[:, "p1p0"].values)

    df_r = df_r.drop(columns="p1p0")
    
    return df_r

def return_calculate_mat(prices, method = "ARITHMETIC", dateCol = "Date"):
    if (dateCol != "NULL"):
        date = prices[dateCol][1:]
        prices = prices.drop(dateCol, axis = 1)
         
    if(method == "BM"):
        df_r = prices.diff()[1:].copy()

    #Arithmetic Return
    if(method == "ARITHMETIC"):
        df_r = prices.pct_change()[1:].copy()
        # the pct_change() function is verified to be performing expected Arithmetic Return calculation
        # prices.pct_change() == (prices/prices.shift(1) - 1) 
        
    #Geometric Brownian Motion
    if(method == "GBM"):
        df_r = np.log(prices/prices.shift(1))[1:].copy()
    
    if (dateCol != "NULL"):
        df_r = pd.concat([date, df_r], axis = 1)
    
        
    return df_r