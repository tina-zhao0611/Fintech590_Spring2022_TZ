import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 

#AR(1) Model
n = 1000  
y_t_1_AR1 = [1]
yt_AR1 = []
for i in range(0, n):
    y = y_t_1_AR1[i] * 0.5 + np.random.randn(1)[0]*0.1
    y_t_1_AR1.append(y)
    yt_AR1.append(y)

#AR(2) Model
y_t_2_AR2 = [0.5, 1]
y_t_1_AR2 = [1]
yt_AR2 = []
for i in range(0, n):
    y = y_t_1_AR2[i] * 0.7 - y_t_2_AR2[i] * 0.4 + np.random.randn(1)[0]*0.1
    y_t_2_AR2.append(y)
    y_t_1_AR2.append(y)
    yt_AR2.append(y)

#AR(3) Model
y_t_3_AR3 = [1.1, 0.5, 1]
y_t_2_AR3 = [0.5, 1]
y_t_1_AR3 = [1]
yt_AR3 = []
for i in range(0, n):
    y = y_t_1_AR3[i] * 0.3 + y_t_2_AR3[i] * 0.2 + y_t_3_AR3[i] * 0.1 + np.random.randn(1)[0]*0.1
    y_t_3_AR3.append(y)
    y_t_2_AR3.append(y)
    y_t_1_AR3.append(y)
    yt_AR3.append(y)

plt.cla()
plt.plot(yt_AR1)
plt.savefig("Week02\\Problem3_AR1.png")
plt.cla()
plot_acf(pd.DataFrame({'y_t': np.transpose(yt_AR1)}), title = "ACF----AR1",lags = 10)
plt.savefig("Week02\\Problem3_AR1_ACF.png")
plt.cla()
plot_pacf(pd.DataFrame({'y_t': np.transpose(yt_AR1)}), title = "PACF----AR1",lags = 10, method='ywm')
plt.savefig("Week02\\Problem3_AR1_PACF.png")

plt.cla()
plt.plot(yt_AR2)
plt.savefig("Week02\\Problem3_AR2.png")
plt.cla()
plot_acf(pd.DataFrame({'y_t': np.transpose(yt_AR2)}), title = "ACF----AR2",lags = 10)
plt.savefig("Week02\\Problem3_AR2_ACF.png")
plt.cla()
plot_pacf(pd.DataFrame({'y_t': np.transpose(yt_AR2)}), title = "PACF----AR2", lags = 10, method='ywm')
plt.savefig("Week02\\Problem3_AR2_PACF.png")

plt.cla()
plt.plot(yt_AR3)
plt.savefig("Week02\\Problem3_AR3.png")
plt.cla()
plot_acf(pd.DataFrame({'y_t': np.transpose(yt_AR3)}), title = "ACF----AR3",lags = 10)
plt.savefig("Week02\\Problem3_AR3_ACF.png")
plt.cla()
plot_pacf(pd.DataFrame({'y_t': np.transpose(yt_AR3)}), title = "PACF----AR3", lags = 10, method='ywm')
plt.savefig("Week02\\Problem3_AR3_PACF.png")

#MA(1) Model
et_1_MA1 = []
et_1_MA1.append(np.random.randn(1)[0])
et_MA1 = []
y_MA1 = []
for i in range(0, n):
    e_t = np.random.randn(1)[0]
    y = et_1_MA1[i] * 0.5 + e_t
    et_1_MA1.append(e_t)
    et_MA1.append(e_t)
    y_MA1.append(y)

#MA(2) Model
et_2_MA2 =[]
et_2_MA2.append(np.random.randn(1)[0])
et_2_MA2.append(np.random.randn(1)[0])
et_1_MA2 = [et_2_MA2[1]]
et_MA2 = []
y_MA2 = []
for i in range(0, n):
    e_t = np.random.randn(1)[0]
    y = et_2_MA2[i] * 0.2 + et_1_MA2[i] * 0.5 + e_t
    et_2_MA2.append(e_t)
    et_1_MA2.append(e_t)
    et_MA2.append(e_t)
    y_MA2.append(y)

#MA(3) Model
et_3_MA3 =[]
et_3_MA3.append(np.random.randn(1)[0])
et_3_MA3.append(np.random.randn(1)[0])
et_3_MA3.append(np.random.randn(1)[0])
et_2_MA3 = [et_3_MA3[1], et_3_MA3[2]]
et_1_MA3 = [et_3_MA3[2]]
et_MA3 = []
y_MA3 = []
for i in range(0, n):
    e_t = np.random.randn(1)[0]
    y = -et_3_MA3[i] * 0.2 + et_2_MA3[i] * 0.5 + et_1_MA3[i] * 0.7 + e_t
    et_3_MA3.append(e_t)
    et_2_MA3.append(e_t)
    et_1_MA3.append(e_t)
    et_MA3.append(e_t)
    y_MA3.append(y)

plt.cla()
plt.plot(y_MA1)
plt.savefig("Week02\\Problem3_MA1.png")
plt.cla()
plot_acf(pd.DataFrame({'y_t': np.transpose(y_MA1)}), title = "ACF----MA1",lags = 10)
plt.savefig("Week02\\Problem3_MA1_ACF.png")
plt.cla()
plot_pacf(pd.DataFrame({'y_t': np.transpose(y_MA1)}), title = "PACF----MA1",lags = 10, method='ywm')
plt.savefig("Week02\\Problem3_MA1_PACF.png")

plt.cla()
plt.plot(y_MA2)
plt.savefig("Week02\\Problem3_MA2.png")
plt.cla()
plot_acf(pd.DataFrame({'y_t': np.transpose(y_MA2)}), title = "ACF----MA2",lags = 10)
plt.savefig("Week02\\Problem3_MA2_ACF.png")
plt.cla()
plot_pacf(pd.DataFrame({'y_t': np.transpose(y_MA2)}), title = "PACF----MA2",lags = 10, method='ywm')
plt.savefig("Week02\\Problem3_MA2_PACF.png")

plt.cla()
plt.plot(y_MA3)
plt.savefig("Week02\\Problem3_MA3.png")
plt.cla()
plot_acf(pd.DataFrame({'y_t': np.transpose(y_MA3)}), title = "ACF----MA3",lags = 10)
plt.savefig("Week02\\Problem3_MA3_ACF.png")
plt.cla()
plot_pacf(pd.DataFrame({'y_t': np.transpose(y_MA3)}), title = "PACF----MA3",lags = 10, method='ywm')
plt.savefig("Week02\\Problem3_MA3_PACF.png")