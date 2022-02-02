import pandas as pd
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt

#calculate the weights vector with n elements, passing in nambda
def expWeight(n, nambda):
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = (1 - nambda) * pow(nambda, n -i + 1) 
    #in the input data, the latest value is at the bottom
    #so generating weights accordingly (largest weight should be at the back)
    weights = weights / sum(weights)
    return weights

#calculate the weighted covariance between two variables
def cov_2arrays(x, y, w):
    cov = 0;
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    n = len(x)
    for i in range(n):
        cov += w[i] * (x[i] - mean_x) * (y[i] - mean_y)
    return cov

#getting covariance matrix (df is the dataframe holding the data)
def expWeightedCov(df, nambda):
    n, m = df.shape
    weights = expWeight(n, nambda)
    Cov = pd.DataFrame(index = df.columns[1:], columns = df.columns[1:]) #the first column is the date
    for i in range(m - 1):
        for j in range(m - 1):
            Cov.iloc[i][j] = cov_2arrays(df.iloc[:, i + 1], df.iloc[:, j + 1], weights)
    return Cov


#reading the input data
data = pd.read_csv("Week03\\DailyReturn.csv")

def pcaGraph(nambda):
    cov = expWeightedCov(data, nambda)
    
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
    plt.savefig("Week03\\Plots\\Problem1_λ = "+ str(nambda) +".png")

# changing the value of λ
pcaGraph(0.3)
pcaGraph(0.5)
pcaGraph(0.8)
pcaGraph(0.95)
pcaGraph(0.97)  
pcaGraph(0.99)  