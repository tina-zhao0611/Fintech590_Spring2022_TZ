import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import minimize

data = pd.read_csv("Week02\\problem2.csv")
x = np.array(data["x"])
y = np.array(data["y"])

plt.figure(figsize = (8,5))
plt.xlabel("x") 
plt.ylabel("y")
plt.scatter(x, y)
plt.title("XY Scatter")
plt.savefig("Week02\\Problem2_Scatter.png")

regression = sm.OLS(y, sm.add_constant(x)) 
model_ols = regression.fit() 
data["yhat"] = model_ols.params[0] + model_ols.params[1]*data["x"]
data["resid"] = model_ols.resid
print(model_ols.summary())

plt.cla()
data.plot(x="x", y="y",kind="scatter",figsize=(8,5))
plt.plot(data["x"], model_ols.params[0] + model_ols.params[1]*data["x"],"r")
plt.text(-2, 4, "y="+ str(round(model_ols.params[0],3)) + "+" +str(round(model_ols.params[1],3)) + "*x" )
plt.title("OLS Fitted")
plt.savefig("Week02\\Problem2_OLS_fitted.png")

plt.cla()
plt.figure(figsize = (8,5))
plt.xlabel("x") 
plt.ylabel("residual")
plt.scatter(x, data["resid"])
plt.title("Resodual-X Scatter")
plt.savefig("Week02\\Problem2_x_residualScatter.png")
plt.figure(figsize = (8,5))
plt.xlabel("yhat") 
plt.ylabel("residual")
plt.scatter(data["yhat"], data["resid"])
plt.title("Resodual-Yhat Scatter")
plt.savefig("Week02\\Problem2_y_residualScatter.png")

plt.cla()
plt.figure(figsize=(8,5))
sns.kdeplot(data["resid"], shade=True, color="g")
plt.title('Density Plot of OLS Residual')
plt.savefig("Week02\\Problem2_residualDensity.png")
#little bit negetive skew

sm.qqplot(data.resid, line='s')
plt.savefig("Week02\\Problem2_residual_QQplot.png")

#MLE -- assume mormality
def likelyhood_norm(parameters): 
    c = parameters[0] 
    b = parameters[1] 
    
    yhat_mle = b * x + c 
        
    L = np.sum(np.log(st.norm.pdf(y - yhat_mle, loc = 0)))
    return -L
mle_model_norm = minimize(likelyhood_norm, np.array([0.1, 0.6]))

#MLE -- assume T distribution
def likelyhood_t(parameters): 
    c = parameters[0] 
    b = parameters[1] 
    
    yhat_mle = b * x + c 
        
    L = np.sum(np.log(st.t.pdf(y - yhat_mle, len(x) - 2)))
    return -L

mle_model_t = minimize(likelyhood_t, np.array([0.1, 0.6]))

plt.cla()
print("OLS: ", model_ols.params, "\nMLE-Assume normality:", mle_model_norm.x, "Log Likelihood: ", -mle_model_norm.fun, "\nMLE-Assume T distribution: ", mle_model_t.x, "Log Likelihood: ", -mle_model_t.fun,) 