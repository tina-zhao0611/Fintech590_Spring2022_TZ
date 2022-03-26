'''
module for valuing options using binomial tree
'''

import numpy as np

if __name__ == '__main__':
    K = 165
    S0 = 165
    exp_date = "04/15/2022"
    r_benefit = 0.0053


    sigma = 0.3
    current_date = "03/13/2022"
    rf = 0.0025
    
    u = np.exp(sigma * np.sqrt(t))