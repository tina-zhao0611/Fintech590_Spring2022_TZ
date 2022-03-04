'''
a option class

initialize a option contract

getT: get time to maturity, return a float in years
valueBS: using BS function method to calculate option value
getImpVol: solve for implied volatility when passing in real market price

'''

import datetime
import numpy as np
import scipy.stats as st
from scipy.optimize import fsolve


class option:
    def __init__(self, type,  exp_date, K, S0, r_benefit = 0, r_cost = 0, dateformat = '%m/%d/%Y'):
        self.type = type.lower()
        self.exp_date = datetime.datetime.strptime(exp_date, dateformat)
        self.K = K
        self.S0 = S0
        self.r_benefit = r_benefit
        self.r_cost = r_cost

    #calcuate time to maturity(calendar days)        
    def getT(self, current_date, dateformat = '%m/%d/%Y', daysForward = 0):  # adding "daysForward" for changing time to maturity when simulatin
        current_date = datetime.datetime.strptime(current_date, dateformat)
        T = ((self.exp_date - current_date).days - daysForward) / 365
        return T
    
    def resetUnerlyingValue(self, underlyingValue):
        self.S0 = underlyingValue
    
    #get option value using BSM function
    def valueBS(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0): 
        # adding "daysForward" for changing time to maturity when simulatin
        T = self.getT(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
        r = rf
        b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
        d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(self.S0 / self.K) + (b - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if self.type == 'call':
            value = self.S0 * np.exp((b - r) * T) * st.norm.cdf(d1, 0, 1) - self.K * np.exp(-r * T) * st.norm.cdf(d2, 0, 1)
        elif self.type == 'put':
            value = self.K * np.exp(-r * T) * st.norm.cdf(-d2, 0, 1) - self.S0 * np.exp((b - r) * T) * st.norm.cdf(-d1, 0, 1)
        return value
    
        
    def getImpVol(self, current_date, rf, value, dateformat = '%m/%d/%Y', daysForward = 0):
        # value is the real market price of option
        
        def vol_helper(vol): #solve this function for zero to get implied volatility
            result = value - self.valueBS(current_date = current_date, sigma = vol, rf = rf, dateformat = dateformat, daysForward = daysForward) - 0.0001
            return result
   
        impliedVol = fsolve(vol_helper, 0.3)
        return impliedVol


#test for the model
if __name__ == '__main__':
    test = option(type = "put", exp_date = "03/18/2022", K = 165, S0 = 185, r_benefit = 0.0056, r_cost=0, dateformat='%m/%d/%Y')
    value = test.valueBS(current_date="02/25/2022", sigma = 0.4, rf = 0.0025)
    print("value = ", value)
    VOL = test.getImpVol(current_date="02/25/2022", rf = 0.0025, value = value)
    print("VOL = ", VOL) # should be 0.4
    test.resetUnerlyingValue(190)
    VOL = test.getImpVol(current_date="02/25/2022", rf = 0.0025, value = value)
    print("VOL = ", VOL)
