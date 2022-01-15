import pandas as pd
import numpy as np
import statsmodels.api as sm

data = pd.read_csv("Week02\\problem1.csv")
x = np.array(data["x"])
y = np.array(data["y"])
df = pd.DataFrame({'x': x, 'y': y})

#get OLS estimated parameters
regression = sm.OLS(y, sm.add_constant(x)) 
model_ols = regression.fit() 
print(model_ols.summary())

Matrix_cov = np.cov(x, y)
cov_xy = Matrix_cov[0][1]
var_x = Matrix_cov[0][0]
u_x = x.mean()
u_y = y.mean()

#there exists floating errors, rounding them to 8 desimal point won't affect the comparison 
df["yhat_OLS"] = round(model_ols.params[0] + model_ols.params[1] * df["x"], 8)
df["yhat_multinorm"] = u_y + (cov_xy * (x - u_x) / var_x)
df["yhat_multinorm"] = round(df["yhat_multinorm"], 8)
 

#compare the two estimates
print(df.loc[lambda x:x['yhat_multinorm']>x['yhat_OLS']])
# resulted in a empty DataFrame, which means the two are the same

