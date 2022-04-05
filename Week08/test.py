from symtable import Symbol
import sys,os

from sympy import div
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from RiskMgmnt import Options, americanBT
# def __init__(self, type,  exp_date, K, S0, r_benefit = 0, r_cost = 0, dateformat = '%m/%d/%Y'):

K = 165
S0 = 165
exp_date = "04/15/2022"
# r_benefit = 0.0053
r_benefit = 0

# European options
opt_call = Options.option("call", exp_date, K, S0, r_benefit)
opt_put = Options.option("put", exp_date, K, S0, r_benefit)


sigma = 0.2
current_date = "03/13/2022"
rf = 0.0025
# print(opt_put.getT(current_date))
# print(opt_call.valueBS(current_date, sigma, rf))
# print(opt_put.valueBS(current_date, sigma, rf))


call_gbsm = opt_call.getAllGreeks(current_date, sigma, rf, method = "GBSM")
call_fd_c = opt_call.getAllGreeks(current_date, sigma, rf, method = "FD")
call_fd_f = opt_call.getAllGreeks(current_date, sigma, rf, method = "FD", how = "FORWARD")
call_fd_b = opt_call.getAllGreeks(current_date, sigma, rf, method = "FD", how = "BACKWARD")

put_gbsm = opt_put.getAllGreeks(current_date, sigma, rf, method = "GBSM")
put_fd_c = opt_put.getAllGreeks(current_date, sigma, rf, method = "FD")
put_fd_f = opt_put.getAllGreeks(current_date, sigma, rf, method = "FD", how = "FORWARD")
put_fd_b = opt_put.getAllGreeks(current_date, sigma, rf, method = "FD", how = "BACKWARD")

ls = [pd.DataFrame(call_gbsm, index = ["Call(GBSM)"]).T, 
                       pd.DataFrame(call_fd_c, index = ["Call(FD_central)"]).T, 
                       pd.DataFrame(call_fd_f, index = ["Call(FD_forward)"]).T, 
                       pd.DataFrame(call_fd_b, index = ["Call(FD_backward)"]).T, 
                       pd.DataFrame(put_gbsm, index = ["Put(GBSM)"]).T, 
                       pd.DataFrame(put_fd_c, index = ["Put(FD_central)"]).T, 
                       pd.DataFrame(put_fd_f, index = ["Put(FD_forward)"]).T, 
                       pd.DataFrame(put_fd_b, index = ["Put(FD_backward)"]).T,]
df_greeks = pd.concat(ls, axis = 1)
print(df_greeks)

# American options
# without dividen payment
ttm = opt_call.getT(current_date)
v_american_call = americanBT.americanBT(opt_call, ttm, rf, sigma, steps = 33, div = [], divT = [])
v_american_put = americanBT.americanBT(opt_put, ttm, rf, sigma, steps = 33, div = [], divT = [])


# with dividend payment
v_american_call_div = americanBT.americanBT(opt_call, ttm, rf, sigma, steps = 33, div = [0.88], divT = [29])
v_american_put_div = americanBT.americanBT(opt_put, ttm, rf, sigma, steps = 33, div = [0.88], divT = [29])

print("Value of American call without dividend payment: ", v_american_call)
print("Value of American call with dividend payment: ", v_american_call_div)
print("Value of American put without dividend payment: ", v_american_put)
print("Value of American put with dividend payment: ", v_american_put_div)

call_fd_c_a = americanBT.getAllGreeks(opt_call, current_date, sigma, rf, steps = 33, div = [], divT = [])
call_fd_c_a_div = americanBT.getAllGreeks(opt_call, current_date, sigma, rf, steps = 33, div = [0.88], divT = [29])
# call_fd_f_a = americanBT.getAllGreeks(opt_call, current_date, sigma, rf, steps = 33, div = [], divT = [], how = "FORWARD")
# call_fd_b_a = americanBT.getAllGreeks(opt_call, current_date, sigma, rf, steps = 33, div = [], divT = [], how = "BACKWARD")

put_fd_c_a = americanBT.getAllGreeks(opt_put, current_date, sigma, rf, steps = 33, div = [], divT = [])
put_fd_c_a_div = americanBT.getAllGreeks(opt_put, current_date, sigma, rf, steps = 33, div = [0.88], divT = [29])
# put_fd_f_a = americanBT.getAllGreeks(opt_put, current_date, sigma, rf, steps = 33, div = [], divT = [], how = "FORWARD")
# put_fd_b_a = americanBT.getAllGreeks(opt_put, current_date, sigma, rf, steps = 33, div = [], divT = [], how = "BACKWARD")

ls_a = [pd.DataFrame(call_fd_c_a, index = ["American Call(no dividend)"]).T, 
        pd.DataFrame(call_fd_c_a_div, index = ["American Call(with dividend)"]).T, 
        pd.DataFrame(put_fd_c_a, index = ["American Put(no dividend)"]).T, 
        pd.DataFrame(put_fd_c_a_div, index = ["American Put(with dividend)"]).T,]
df_greeks_a = pd.concat(ls_a, axis = 1)
print(df_greeks_a)

# examine the sensitivity in the dividend amount
dividend = np.linspace(0,1)
vAmCall = []
vAmPut = []
for div in dividend:
    vAmCall.append (americanBT.americanBT(opt_call, ttm, rf, sigma, steps = 33, div = [div], divT = [29]))
    vAmPut.append (americanBT.americanBT(opt_put, ttm, rf, sigma, steps = 33, div = [div], divT = [29]))
    
plt.subplot(1, 2, 1)    
plt.plot(dividend, vAmCall, linewidth=2, label = "American call")
plt.ylim(3, 4.5) #set the scale to make the two plots comparable
plt.ylabel("option value")
plt.xlabel("dividend amount")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(dividend, vAmPut, linewidth=2, label = "American put")
plt.ylim(3.5, 5)
plt.xlabel("dividend amount")
plt.legend()

plt.suptitle("Sensitivity Comparision")
plt.savefig("Week07\\plots\\Problem1_dividendSensitivity.png")


# changing the dividend payment time
dividend = np.linspace(0,1)
vAmCall = []
vAmPut = []
for div in dividend:
    vAmCall.append (americanBT.americanBT(opt_call, ttm, rf, sigma, steps = 33, div = [div], divT = [5]))
    vAmPut.append (americanBT.americanBT(opt_put, ttm, rf, sigma, steps = 33, div = [div], divT = [5]))

plt.subplot(1, 2, 1)
plt.cla() 
plt.plot(dividend, vAmCall, linewidth=2, label = "American call")
plt.ylim(3, 4.5) #set the scale to make the two plots comparable
plt.ylabel("option value")
plt.xlabel("dividend amount")
plt.legend()

plt.subplot(1, 2, 2)
plt.cla()
plt.plot(dividend, vAmPut, linewidth=2, label = "American put")
plt.ylim(3.5, 5)
plt.xlabel("dividend amount")
plt.legend()

plt.suptitle("Sensitivity Comparision (dividend paid earlier)")
plt.savefig("Week07\\plots\\Problem1_dividendSensitivity2.png")