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
        self.exp_date = exp_date
        self.K = K
        self.S0 = S0
        self.r_benefit = r_benefit
        self.r_cost = r_cost

    #calcuate time to maturity(calendar days)        
    def getT(self, current_date, dateformat = '%m/%d/%Y', daysForward = 0):  # adding "daysForward" for changing time to maturity when simulatin
        current = datetime.datetime.strptime(current_date, dateformat)
        exp = datetime.datetime.strptime(self.exp_date, dateformat)
        T = ((exp - current).days - daysForward) / 365
        return T
    
    def resetUnerlyingValue(self, underlyingValue):
        self.S0 = underlyingValue
    
    #get option value using BSM function
    def valueBS(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, T = 9999): 
        # adding "daysForward" for changing time to maturity when simulatin
        if (T == 9999):
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

    def Delta(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if(method == "GBSM"):
            T = self.getT(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            if(self.type == "call"):
                delta = np.exp((b - r) * T) * st.norm.cdf(d1, 0, 1)
            elif(self.type == "put"):
                delta = np.exp((b - r) * T) * (st.norm.cdf(d1, 0, 1) - 1)
        elif(method == "FD"):
            d = 1e-8
            if(how == "CENTRAL"):
                # d = sys.float_info.min
                opt_up = option(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0 + d, r_benefit = self.r_benefit, r_cost = self.r_cost, dateformat = dateformat)
                vup = opt_up.valueBS(current_date, sigma, rf, dateformat, daysForward)
                opt_down = option(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0 - d, r_benefit = self.r_benefit, r_cost = self.r_cost, dateformat = dateformat)
                vdown = opt_down.valueBS(current_date, sigma, rf, dateformat, daysForward)
                delta = (vup - vdown)/(2 * d)
            elif(how == "FORWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                opt_up = option(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
                vup = opt_up.valueBS(current_date, sigma, rf, dateformat, daysForward)
                delta = (vup- v)/ d
            elif(how == "BACKWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                opt_down = option(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
                vdown = opt_down.valueBS(current_date, sigma, rf, dateformat, daysForward)
                delta = (v - vdown)/ d
        return delta
    
    def Gamma(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if(method == "GBSM"):
            T = self.getT(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            gamma = np.exp((b - r) * T) * st.norm.pdf(d1, 0, 1) / (self.S0 * sigma * np.sqrt(T))
        elif(method == "FD"):
            d = 1e-8
            if(how == "CENTRAL"):
                # d = sys.float_info.min
                opt_up = option(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
                deltaUp = opt_up.Delta(current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM")
                opt_down = option(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
                deltaDown = opt_down.Delta(current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM")
                gamma = (deltaUp - deltaDown)/(2 * d)
            elif(how == "FORWARD"):
                delta = self.Delta(current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM")
                opt_up = option(self.type, self.exp_date, self.K, self.S0 + d, self.r_benefit, self.r_cost, dateformat)
                deltaUp = opt_up.Delta(current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM")
                gamma = (deltaUp- delta)/ d
            elif(how == "BACKWARD"):
                delta = self.Delta(current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM")
                opt_down = option(self.type, self.exp_date, self.K, self.S0 - d, self.r_benefit, self.r_cost, dateformat)
                deltaDown = opt_down.Delta(current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM")
                gamma = (delta - deltaDown)/ d
        return gamma
    
    def Vega(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if(method == "GBSM"):
            T = self.getT(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            vega = self.S0 * np.exp((b - r) * T) * st.norm.pdf(d1, 0, 1) * np.sqrt(T)
        elif(method == "FD"):
            d = 1e-8
            if(how == "CENTRAL"):
                # d = sys.float_info.min
                vUp = self.valueBS(current_date, sigma + d, rf, dateformat, daysForward)
                vDown = self.valueBS(current_date, sigma - d, rf, dateformat, daysForward)
                vega = (vUp - vDown)/(2 * d)
            elif(how == "FORWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                vUp = self.valueBS(current_date, sigma + d, rf, dateformat, daysForward)
                vega = (vUp- v)/ d
            elif(how == "BACKWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                vDown = self.valueBS(current_date, sigma - d, rf, dateformat, daysForward)
                vega = (v - vDown)/ d
        return vega

    def Theta(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if(method == "GBSM"):
            T = self.getT(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = (np.log(self.S0 / self.K) + (b - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            if(self.type == "call"):
                theta = -(self.S0 * np.exp((b - r) * T) * st.norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T))) - (b - r) * self.S0 * np.exp((b - r) * T) * st.norm.cdf(d1, 0, 1) - r * self.K * np.exp(-r * T) * st.norm.cdf(d2, 0, 1)
            elif(self.type == "put"):
                theta = -(self.S0 * np.exp((b - r) * T) * st.norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T))) + (b - r) * self.S0 * np.exp((b - r) * T) * st.norm.cdf(-d1, 0, 1) + r * self.K * np.exp(-r * T) * st.norm.cdf(-d2, 0, 1)
        elif(method == "FD"):
            d = 1e-8
            T = self.getT(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            if(how == "CENTRAL"):
                # d = sys.float_info.min
                vUp = self.valueBS(current_date, sigma, rf, dateformat, daysForward, T = T - d)
                vDown = self.valueBS(current_date, sigma, rf, dateformat, daysForward, T = T + d)
                theta = (vUp - vDown)/(2 * d)
            elif(how == "FORWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                vUp = self.valueBS(current_date, sigma, rf, dateformat, daysForward, T = T - d)
                theta = (vUp- v)/ d
            elif(how == "BACKWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                vDown = self.valueBS(current_date, sigma, rf, dateformat, daysForward, T = T + d)
                theta = (v - vDown)/ d
        return -theta

    def Rho(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if(method == "GBSM"):
            T = self.getT(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d2 = (np.log(self.S0 / self.K) + (b - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            if(self.type == "call"):
                rho = T * self.K * np.exp(-r * T) * st.norm.cdf(d2, 0, 1)
            elif(self.type == "put"):
                rho = -T * self.K * np.exp(-r * T) * st.norm.cdf(-d2, 0, 1)
        elif(method == "FD"):
            d = 1e-8
            if(how == "CENTRAL"):
                # d = sys.float_info.min
                vUp = self.valueBS(current_date, sigma, rf + d, dateformat, daysForward)
                vDown = self.valueBS(current_date, sigma, rf - d, dateformat, daysForward)
                rho = (vUp - vDown)/(2 * d)
            elif(how == "FORWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                vUp = self.valueBS(current_date, sigma, rf + d, dateformat, daysForward)
                rho = (vUp- v)/ d
            elif(how == "BACKWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                vDown = self.valueBS(current_date, sigma, rf - d, dateformat, daysForward)
                rho = (v - vDown)/ d
        return rho
    
    def CarryRho(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        if(method == "GBSM"):
            T = self.getT(current_date = current_date, dateformat = dateformat, daysForward = daysForward)
            r = rf
            b = r - self.r_benefit + self.r_cost  # b = r - q when there is a continues dividend/coupon rate
            d1 = (np.log(self.S0 / self.K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            if(self.type == "call"):
                carryrho = T * self.S0 * np.exp((b - r) * T) * st.norm.cdf(d1, 0, 1)
            elif(self.type == "put"):
                carryrho = -T * self.S0 * np.exp((b - r) * T) * st.norm.cdf(-d1, 0, 1)
        elif(method == "FD"):
            d = 1e-8
            if(how == "CENTRAL"):
                # d = sys.float_info.min
                opt_up = option(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit - d, r_cost = self.r_cost, dateformat = dateformat)
                vUp = opt_up.valueBS(current_date, sigma, rf, dateformat, daysForward)
                opt_down = option(type = self.type, exp_date = self.exp_date, K = self.K, S0 = self.S0, r_benefit = self.r_benefit + d, r_cost = self.r_cost, dateformat = dateformat)
                vDown = opt_down.valueBS(current_date, sigma, rf, dateformat, daysForward)
                carryrho = (vUp - vDown)/(2 * d)
            elif(how == "FORWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                opt_up = option(self.type, self.exp_date, self.K, self.S0, self.r_benefit - d, self.r_cost, dateformat)
                vUp = opt_up.valueBS(current_date, sigma, rf, dateformat, daysForward)
                carryrho = (vUp- v)/ d
            elif(how == "BACKWARD"):
                v = self.valueBS(current_date, sigma, rf, dateformat, daysForward)
                opt_down = option(self.type, self.exp_date, self.K, self.S0, self.r_benefit + d, self.r_cost, dateformat)
                vDown = opt_down.valueBS(current_date, sigma, rf, dateformat, daysForward)
                carryrho = (v - vDown)/ d
        return carryrho

    def getAllGreeks(self, current_date, sigma, rf, dateformat = '%m/%d/%Y', daysForward = 0, method = "GBSM", how = "CENTRAL"):
        greeks = {}
        greeks["delta"] = self.Delta(current_date, sigma, rf, dateformat, daysForward, method, how)
        greeks["gamma"] = self.Gamma(current_date, sigma, rf, dateformat, daysForward, method, how)
        greeks["theta"] = self.Theta(current_date, sigma, rf, dateformat, daysForward, method, how)
        greeks["vega"] = self.Vega(current_date, sigma, rf, dateformat, daysForward, method, how)
        greeks["rho"] = self.Rho(current_date, sigma, rf, dateformat, daysForward, method,how)
        greeks["carryrho"] = self.CarryRho(current_date, sigma, rf, dateformat, daysForward, method, how)
        return greeks
    
#test for the model
if __name__ == '__main__':
    test = option(type = "call", exp_date = "03/18/2022", K = 165, S0 = 185, r_benefit = 0.0056, r_cost=0, dateformat='%m/%d/%Y')
    value = test.valueBS(current_date="02/25/2022", sigma = 0.4, rf = 0.0025)
    print("value = ", value)
    VOL = test.getImpVol(current_date="02/25/2022", rf = 0.0025, value = value)
    print("VOL = ", VOL) # should be 0.4
    test.resetUnerlyingValue(190)
    VOL = test.getImpVol(current_date="02/25/2022", rf = 0.0025, value = value)
    print("VOL = ", VOL)
    d = test.Theta(current_date="02/25/2022", sigma = 0.4, rf = 0.0025)
    print(d)
    d = test.Theta(current_date="02/25/2022", sigma = 0.4, rf = 0.0025, method = "FD")
    print(d)
    
    current_price = 164.85
    rf = 0.0025
    r_benefit = 0.0053
    current_date = "02/25/2022"
    test1 = option("call", "03/18/2022", 165, current_price, r_benefit)
    print(test1.getImpVol(current_date, rf, 4.5))
    print(test1.getT(current_date, daysForward=10))