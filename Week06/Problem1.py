import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from RiskMgmnt import Options
# def __init__(self, type,  exp_date, K, S0, r_benefit = 0, r_cost = 0, dateformat = '%m/%d/%Y'):

Kc = 175 # strike price of the call option
Kp = 155 # strike price of the put option
my_call = Options.option("call", exp_date="03/18/2022", K = Kc, r_benefit=0.0053, S0 = 165)
my_put = Options.option("put", exp_date="03/18/2022", K = Kp, r_benefit=0.0053, S0 = 165)

T = my_call.getT(current_date="02/25/2022")
print("Time to maturity is:{:1.3f}Years".format(T))

d_call = {}
d_put = {}
current_date="02/25/2022"
rf = 0.0025
for vol in np.arange (0.1, 0.8, 0.01):
    d_call[vol] = my_call.valueBS(current_date=current_date, sigma = vol, rf = rf)
    d_put[vol] = my_put.valueBS(current_date=current_date, sigma = vol, rf = rf)
    
df_call = pd.DataFrame(d_call, index = ["value_call"]).T
df_put = pd.DataFrame(d_put, index = ["value_put"]).T

data = df_call.join(df_put)

#plot the result
plt.cla()
plt.plot(data.index, data["value_call"], label = "call(K = " + str(Kc) + ")")
plt.plot(data.index, data["value_put"], color = 'r', label = "put(K = " + str(Kp) + ")")
plt.xlabel("implied vol")
plt.ylabel("value")
plt.legend()
plt.title("option value (S0 = 165)")
plt.savefig("Week06\\plots\\Problem1.png")