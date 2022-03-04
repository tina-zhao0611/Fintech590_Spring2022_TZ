import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import matplotlib.pyplot as plt


from RiskMgmnt import Options
# def __init__(self, type,  exp_date, K, S0, r_benefit = 0, r_cost = 0, dateformat = '%m/%d/%Y'):
    
df_options = pd.read_csv("Week06\\AAPL_Options.csv")
col_names = df_options.columns.tolist()
    # there's a blank space in the column name 
    # that cause the "itertuples" to return wrong attribute name
for index,value in enumerate(col_names):
    col_names[index]= value.replace(" ","")
df_options.columns=col_names 

#data
current_price = 164.85
rf = 0.0025
r_benefit = 0.0053
current_date = "02/25/2022"

def getVol(df):
    implied_vol = {}
    for row in df.itertuples(name = "options"):
        opt = Options.option(type = getattr(row, "Type"), 
                    exp_date = getattr(row, "Expiration"), 
                    K = getattr(row, "Strike"),
                    S0 = current_price,
                    r_benefit = r_benefit)
        
        imply_vol = opt.getImpVol(current_date = current_date, rf = rf, value =  getattr(row, "LastPrice"))
        implied_vol[getattr(row, "Strike")] = imply_vol[0]
    df_vol = pd.DataFrame(list(implied_vol.items()), columns = ["Strike", "ImpliedVolatilities"])
    result = pd.merge(df, df_vol, on = "Strike")
    return result

gp = df_options.groupby("Type")
call = getVol(gp.get_group("Call"))
put = getVol(gp.get_group("Put"))



#plot volatility against strike price
plt.cla()
plt.plot(call["Strike"], call["ImpliedVolatilities"])
plt.axvline(x = current_price, ls = ':',color = 'r', label = "current market price")
plt.xlabel("Strike price")
plt.ylabel("Imlied volatility")
plt.title("Implied Volatility --- Call Options")
plt.legend()
plt.savefig("Week06\\plots\\Problem2_call")

plt.cla()
plt.plot(put["Strike"], put["ImpliedVolatilities"])
plt.axvline(x = current_price, ls = ':',color = 'r', label = "current market price")
plt.xlabel("Strike price")
plt.ylabel("Imlied volatility")
plt.title("Implied Volatility --- Put Options")
plt.legend()
plt.savefig("Week06\\plots\\Problem2_put")
