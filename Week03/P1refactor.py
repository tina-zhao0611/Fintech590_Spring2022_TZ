import pandas as pd
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import sys,os

from sympy import div
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

from RiskMgmnt import expWeighted


#reading the input data
data = pd.read_csv("Week03\\DailyReturn.csv")

def pcaGraph(nambda):
    cov = expWeighted.expWeightedCov(data, nambda)
    
    eigValue = eigh(np.array(cov,dtype=float))[0]
    
    #keep only positive real value
    x = eigValue.shape[0]
    for i in range (x):
        if (eigValue[i] < 1e-8) or (np.imag(eigValue[i]) != 0) :
            eigValue[i] = 0
    eigValue = np.real(eigValue)
    
    #calculate percentage explained and cumulative percentage
    tot = sum(eigValue)
    var_exp = eigValue/tot
    var_exp.sort()
    var_exp = var_exp[::-1]
    cum_var_exp = np.cumsum(var_exp)
    
    #plotting the explained variance and also the cumulative value
    plt.cla()
    plt.bar(range(1,6), np.array(var_exp[:5]), label = 'individual var')
    for x,y in zip(range(1,6),np.array(var_exp[:5])):
        plt.text(x,y,'%.2f' %y, ha='center',va='bottom')
    plt.step(range(1,6), np.array(cum_var_exp[:5]), where = 'mid', label = 'cumulative var')
    plt.ylabel('variance explained')
    plt.xlabel('principal components')
    plt.legend(loc = 'best')
    plt.title("λ = " + str(nambda) )
    plt.savefig("Week03\\Plots_new\\Problem1_λ = "+ str(nambda) +".png")

# changing the value of λ
pcaGraph(0.3)
pcaGraph(0.5)
pcaGraph(0.8)
pcaGraph(0.95)
pcaGraph(0.97)  
pcaGraph(0.99)  