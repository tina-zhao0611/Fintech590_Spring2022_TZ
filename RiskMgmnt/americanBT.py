'''
model for option valuation using binomial tree

methods for calculating greeks using finite difference
'''

import numpy as np
from scipy.optimize import fsolve


# from RiskMgmnt import Options
from RiskMgmnt import Options
    # def __init__(self, type,  exp_date, K, S0, r_benefit = 0, r_cost = 0, dateformat = '%m/%d/%Y'):
    
def americanBT_noDiv(option, ttm, rf, sigma, steps):
    b = rf - option.r_benefit + option.r_cost
    dt = ttm / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1 - pu
    df = np.exp(-rf * dt)
    if(option.type == "call"):
        z = 1
    else:
        z = -1
        
    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)
    
    def idxFunc(i, j ):
        result = nNodeFunc(j - 1) + i
        return result
    
    nNodes = nNodeFunc(steps)
    
    optValues = np.zeros(nNodes)
    
    for j in range(steps, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = option.S0 * (u ** i) * (d **(j - i)) 
            optValues[idx] = max([0, z * (price - option.K)])
        
            if(j < steps):
                optValues[idx] = max([optValues[idx], df * (pu * optValues[idxFunc(i + 1, j + 1)] + pd * optValues[idxFunc(i, j + 1)]) ])
        
    return optValues[0]

def americanBT(option, ttm, rf, sigma, steps, div = [], divT = []):
    if (len(div) == 0) or (len(divT) == 0):
        return americanBT_noDiv(option, ttm, rf, sigma, steps)
    elif(divT[0] > steps):
        return americanBT_noDiv(option, ttm, rf, sigma, steps)

    b = rf - option.r_benefit + option.r_cost
    dt = ttm / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1 - pu
    df = np.exp(-rf * dt)
    if(option.type == "call"):
        z = 1
    else:
        z = -1
        
    def nNodeFunc(n):
            return int((n + 1) * (n + 2) / 2)
    
    def idxFunc(i, j ):
        result = nNodeFunc(j - 1) + i
        return result
    
    nNodes = nNodeFunc(divT[0])
    
    optValues = np.zeros(nNodes)
    
    for j in range(divT[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = option.S0 * (u ** i) * (d ** (j - i)) 
            if (j < divT[0]):
                optValues[idx] = max([0, z * (price - option.K)])
                optValues[idx] = max([optValues[idx], df * (pu * optValues[idxFunc(i + 1, j + 1)] + pd * optValues[idxFunc(i, j + 1)])])

            else:
                newOpt = Options.option(option.type, option.exp_date, option.K, price - div[0])
                vNE = americanBT(newOpt, ttm - divT[0] * dt, rf, sigma, steps - divT[0], div[1:], divT[1:])
                vE = max([0, z * (price - option.K)])
                optValues[idx] = max([vNE, vE])
    
    return optValues[0]

def getImpVol(option, current_date, rf, value, steps, div, divT):
        # value is the real market price of option
    ttm = option.getT(current_date)
    def vol_helper(vol): #solve this function for zero to get implied volatility
        result = value - americanBT(option, ttm, rf, vol, steps, div, divT) - 0.0001
        return result
   
    impliedVol = fsolve(vol_helper, 0.3)
    return impliedVol

def Delta(option, current_date, sigma, rf, steps, div, divT, dateformat = '%m/%d/%Y', how = "CENTRAL"):
    d = 1e-8
    ttm = option.getT(current_date, dateformat)
    if(how == "CENTRAL"):
        opt_up = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0 + d, r_benefit = option.r_benefit, r_cost = option.r_cost, dateformat = dateformat)
        vUp = americanBT(opt_up, ttm,rf, sigma, steps, div, divT)
        opt_down = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0 - d, r_benefit = option.r_benefit, r_cost = option.r_cost, dateformat = dateformat)
        vDown = americanBT(opt_down, ttm,rf, sigma, steps, div, divT)
        delta = (vUp - vDown)/(2 * d)
    elif(how == "FORWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        opt_up = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0 + d, r_benefit = option.r_benefit, r_cost = option.r_cost, dateformat = dateformat)
        vUp = americanBT(opt_up, ttm,rf, sigma, steps, div, divT)
        delta = (vUp- v)/ d
    elif(how == "BACKWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        opt_down = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0 - d, r_benefit = option.r_benefit, r_cost = option.r_cost, dateformat = dateformat)
        vDown = americanBT(opt_down, ttm,rf, sigma, steps, div, divT)
        delta = (v - vDown)/ d
    return delta

def Gamma(option, current_date, sigma, rf, steps, div, divT, dateformat = '%m/%d/%Y', how = "CENTRAL"):
    d = 1e-7
    ttm = option.getT(current_date, dateformat)
    if(how == "CENTRAL"):
        opt_up = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0 + d, r_benefit = option.r_benefit, r_cost = option.r_cost, dateformat = dateformat)
        deltaUp = Delta(opt_up, current_date, sigma, rf, steps, div, divT)
        opt_down = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0 - d, r_benefit = option.r_benefit, r_cost = option.r_cost, dateformat = dateformat)
        deltaDown = Delta(opt_down, current_date, sigma, rf, steps, div, divT)
        gamma = (deltaUp - deltaDown)/(2 * d)
    elif(how == "FORWARD"):
        delta = Delta(option, current_date, sigma, rf, steps, div, divT)
        opt_up = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0 + d, r_benefit = option.r_benefit, r_cost = option.r_cost, dateformat = dateformat)
        deltaUp = Delta(opt_up, current_date, sigma, rf, steps, div, divT)
        gamma = (deltaUp- delta)/ d
    elif(how == "BACKWARD"):
        delta = Delta(option, current_date, sigma, rf, steps, div, divT)
        opt_down = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0 - d, r_benefit = option.r_benefit, r_cost = option.r_cost, dateformat = dateformat)
        deltaDown = Delta(opt_down, current_date, sigma, rf, steps, div, divT)
        gamma = (delta - deltaDown)/ d
    return gamma

def Vega(option, current_date, sigma, rf, steps, div, divT, dateformat = '%m/%d/%Y', how = "CENTRAL"):
    d = 1e-8
    ttm = option.getT(current_date, dateformat)
    if(how == "CENTRAL"):
        vUp = americanBT(option, ttm, rf, sigma + d, steps, div, divT)
        vDown = americanBT(option, ttm, rf, sigma - d, steps, div, divT)
        vega = (vUp - vDown)/(2 * d)
    elif(how == "FORWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        vUp = americanBT(option, ttm, rf, sigma + d, steps, div, divT)
        vega = (vUp- v)/ d
    elif(how == "BACKWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        vDown = americanBT(option, ttm, rf, sigma - d, steps, div, divT)
        vega = (v - vDown)/ d
    return vega

def Theta(option, current_date, sigma, rf, steps, div, divT, dateformat = '%m/%d/%Y', how = "CENTRAL"):
    d = 1e-8
    ttm = option.getT(current_date, dateformat)
    if(how == "CENTRAL"):
        vUp = americanBT(option, ttm + d, rf, sigma, steps, div, divT)
        vDown = americanBT(option, ttm - d, rf, sigma, steps, div, divT)
        theta = (vUp - vDown)/(2 * d)
    elif(how == "FORWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        vUp = americanBT(option, ttm + d, rf, sigma, steps, div, divT)
        theta = (vUp- v)/ d
    elif(how == "BACKWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        vDown = americanBT(option, ttm - d, rf, sigma, steps, div, divT)
        theta = (v - vDown)/ d
    return -theta


def Rho(option, current_date, sigma, rf, steps, div, divT, dateformat = '%m/%d/%Y', how = "CENTRAL"):
    d = 1e-8
    ttm = option.getT(current_date, dateformat)
    if(how == "CENTRAL"):
        vUp = americanBT(option, ttm, rf + d, sigma, steps, div, divT)
        vDown = americanBT(option, ttm, rf - d, sigma, steps, div, divT)
        rho = (vUp - vDown)/(2 * d)
    elif(how == "FORWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        vUp = americanBT(option, ttm, rf + d, sigma, steps, div, divT)
        rho = (vUp- v)/ d
    elif(how == "BACKWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        vDown = americanBT(option, ttm, rf - d, sigma, steps, div, divT)
        rho = (v - vDown)/ d
    return rho

def CarryRho(option, current_date, sigma, rf, steps, div, divT, dateformat = '%m/%d/%Y', how = "CENTRAL"):
    d = 1e-8
    ttm = option.getT(current_date, dateformat)
    if(how == "CENTRAL"):
        opt_up = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0, r_benefit = option.r_benefit - d, r_cost = option.r_cost, dateformat = dateformat)
        vUp = americanBT(opt_up, ttm,rf, sigma, steps, div, divT)
        opt_down = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0, r_benefit = option.r_benefit + d, r_cost = option.r_cost, dateformat = dateformat)
        vDown = americanBT(opt_down, ttm,rf, sigma, steps, div, divT)
        carryrho = (vUp - vDown)/(2 * d)
    elif(how == "FORWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        opt_up = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0, r_benefit = option.r_benefit - d, r_cost = option.r_cost, dateformat = dateformat)
        vUp = americanBT(opt_up, ttm,rf, sigma, steps, div, divT)
        carryrho = (vUp- v)/ d
    elif(how == "BACKWARD"):
        v = americanBT(option, ttm,rf, sigma, steps, div, divT)
        opt_down = Options.option(option.type, exp_date = option.exp_date, K = option.K, S0 = option.S0, r_benefit = option.r_benefit + d, r_cost = option.r_cost, dateformat = dateformat)
        vDown = americanBT(opt_down, ttm,rf, sigma, steps, div, divT)
        carryrho = (v - vDown)/ d
    return carryrho

def getAllGreeks(option, current_date, sigma, rf, steps, div, divT, dateformat = '%m/%d/%Y', how = "CENTRAL"):
    greeks = {}
    greeks["delta"] = Delta(option, current_date, sigma, rf, steps, div, divT, dateformat, how)
    greeks["gamma"] = Gamma(option, current_date, sigma, rf, steps, div, divT, dateformat, how)
    greeks["theta"] = Theta(option, current_date, sigma, rf, steps, div, divT, dateformat, how)
    greeks["vega"] = Vega(option, current_date, sigma, rf, steps, div, divT, dateformat, how)
    greeks["rho"] = Rho(option, current_date, sigma, rf, steps, div, divT, dateformat, how)
    greeks["carryrho"] = CarryRho(option, current_date, sigma, rf, steps, div, divT, dateformat, how)
    return greeks


if __name__ == '__main__':
    test1 = Options.option("put", "03/21/2022", 100, 100)
    v1 = americanBT(test1, 0.5, 0.08, 0.3, 2)
    test2 = Options.option("call", "03/21/2022", 100, 100)
    current = "01/02/2022"
    ttm = test2.getT(current)
    v2 = americanBT(test2, 0.5, 0.08, 0.3, 100)
    v3 = americanBT(test2, 0.5, 0.08, 0.3, 2, [1], [1])
    v5 = americanBT(test2, 0.5, 0.08, 0.3, 4, [1], [2])

    v4 = americanBT(test2, ttm, 0.08, 0.3, 100, [1], [50])
    vol = getImpVol(test2, current, 0.08, v4, 100, [1], [50])
    print(v1,v2,v3, v4,v5)
    print(vol)
